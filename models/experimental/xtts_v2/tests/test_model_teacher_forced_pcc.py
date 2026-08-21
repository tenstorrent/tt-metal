# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced PCC test for the request path: real text in, GPT latents and waveform compared
against the CPU.

Every other gate either checks one block against a reference or checks that repeats are identical,
so a wrong-but-deterministic device result passes all of them. `generate` samples, though, so its
codes cannot be reproduced independently -- one different draw and the two sides speak different
sentences within a few steps. So this hands the device's OWN codes to the CPU and asks which latents
they imply. Both sides then start from identical inputs (embedding lookups) and only the 30 layers
differ: device bf16, KV-cached, trace-replayed single steps against fp32, uncached, one wide pass.
Causal masking makes those the same function, so they must agree.

Conditioning and the speaker embedding are computed once on device and handed to BOTH sides, so
Blocks 1 and 2 are inputs here rather than results -- test_cond_pcc and test_speaker_pcc own those.

Three axes are crossed, because each selects different device work: prompt length (the prefill
program and the fill_cache split), seed (a different path through the decoder), and reference-clip
length (the conditioning conv shapes). Texts are ordinary prose, one per prefill bucket `generate`
can reach, and never repeated -- repeated input makes the model degenerate. Decode is uncapped, so
each request runs the cache out to the model's stop or the 605-code limit, the occupancy axis BUG-1
lived on.

Each vocoder bucket reached also gets a waveform check, since no other gate drives the vocoder with
a real request's latents; both sides get the DEVICE's latents there, so it isolates the vocoder and
its host glue.

Coverage is asserted, not assumed: trimming the texts or seeds fails the test rather than quietly
checking less. Opens its own device, like demo/, so the real trace-region and L1_SMALL sizes are the
ones under test.
"""
import itertools

import torch

# Eager GPT2 import, before any XTTS checkpoint load — see tests/reference_helpers.py.
from transformers import GPT2Model  # noqa: F401

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.reference.xtts_gpt_ref import build_reference, reference_forward
from models.experimental.xtts_v2.reference.xtts_hifigan_ref import HifiganReference
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import _prefill_tiles
from models.experimental.xtts_v2.tt.ttnn_xtts_model import (
    HOP,
    START_AUDIO_TOKEN,
    XttsV2,
    _fade_out,
    _voc_bucket,
    _voc_input,
    _voc_pad,
    assemble_prompt,
)

TARGET_PCC = 0.999  # the two sides differ only in precision and in how the past is fetched
TARGET_PCC_WAV = 0.99  # same gate as test_vocoder_request_path: identical latents, both vocoders
SEEDS = (0, 1, 2)  # each draw walks a different path through the decoder
CLIP_SECONDS = (6.0, 4.0)  # a clip's LENGTH picks the conditioning conv shapes, not its content
MIN_DEPTH = 900  # some run must take the cache near the model's 1042-position ceiling
# Which vocoder buckets get reached is a sampling outcome, so this only guards against the set
# collapsing -- the buckets covered are named in the result. test_vocoder_request_path gates all 7.
MIN_VOC_BUCKETS = 5

# One text per prefill bucket generate() can reach (prompts of 42..429 tokens -> 64..512 rows).
TEXTS = {
    "tiny": "Good morning.",
    "sentence": "The harbour lights came on one by one as the evening fog rolled in from the bay.",
    "two_lines": "She spent the afternoon rearranging the shelves, and only noticed the rain had "
    "stopped when the last book was finally back in place.",
    "short_para": "The bakery on the corner opens before dawn, and by six the whole street smells of "
    "warm bread. Regulars arrive in a predictable order: the night nurse first, then the taxi driver "
    "who never takes off his hat, and finally the schoolchildren who spend a long time choosing "
    "between two things that cost the same.",
    "para": "Cartography was once a matter of guesswork and rumour. A captain would return with a "
    "sketch, a merchant would contradict it, and the mapmaker would split the difference. Whole "
    "islands appeared on charts for a century because nobody wanted to be the first to erase them. "
    "The habit of drawing what one expected to find, rather than what was actually there, took a "
    "surprisingly long time to break, and it cost a great many ships along the way.",
    "long_para": "The archive occupied four floors of a building designed for something else "
    "entirely, and every year the collection pressed a little harder against the walls. Letters "
    "arrived in shoeboxes. Photographs came loose in envelopes with no dates on them. A retired "
    "engineer donated thirty years of notebooks written in a private shorthand that nobody has yet "
    "deciphered. The archivists worked slowly, because speed and accuracy pull in opposite "
    "directions when the material is fragile and the labels are wrong. Visitors often asked why the "
    "collection had not simply been scanned, as though scanning were a single decision rather than a "
    "thousand small ones about what a document actually is.",
}


def _clip(seconds, seed):  # a reference clip's LENGTH is what picks the conv shapes, not its content
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * 22050), generator=g) * 0.1


def run_teacher_forced_pcc(verbose=True):
    tts = XttsV2()
    try:
        tts.warmup()
        # Both clip lengths up front: a new one compiles and allocates conditioning conv shapes.
        voices = [(secs, tts.compute_voice(_clip(secs, i), 22050)) for i, secs in enumerate(CLIP_SECONDS)]
        gpt, final_norm = build_reference(tts.ckpt_path)
        mel_emb, mel_pos = tts.heads["mel_emb"], tts.heads["mel_pos"]
        grid = tts.mesh_device.compute_with_storage_grid_size()
        n_head, n_cores = tts.decoder.config.n_head, grid.x * grid.y

        voc_ref = HifiganReference()
        scored, wav_scored, no_codes, buckets, deepest = [], [], [], set(), 0
        voc_done = set()
        for (name, text), (secs, voice), seed in itertools.product(TEXTS.items(), voices, SEEDS):
            label = f"{name}/{secs:g}s/seed{seed}"
            wav = tts.generate(text, voice, seed=seed)
            codes, dev = tts.last_generation["codes"], tts.last_generation["latents"]
            prefix = assemble_prompt(tts.tokenizer.encode(text, "en"), voice.gpt_cond_latent, tts.tables)
            P = prefix.shape[1]
            buckets.add(32 * _prefill_tiles(P, n_head, n_cores))
            if not codes:  # rare and legitimate: the first sampled code was STOP (see generate)
                no_codes.append(label)
                continue
            deepest = max(deepest, P + len(codes))
            # Position i is fed code i-1 (START at 0), so the last code is never fed back, and the
            # latent at position i is the one that code i was sampled from.
            mel = (mel_emb[[START_AUDIO_TOKEN] + codes[:-1]] + mel_pos[: len(codes)]).unsqueeze(0)
            _, ref = reference_forward(gpt, final_norm, torch.cat([prefix, mel], dim=1))
            ok, pcc = comp_pcc(ref[:, P:, :], dev, pcc=TARGET_PCC)
            if verbose:
                print(f"  {label:22s} P={P:4d} codes={len(codes):4d} depth={P + len(codes):5d}  pcc: {pcc}")
            scored.append((label, ok, pcc))

            # The vocoder's work is fixed by its length bucket, not by the text or the seed, so
            # check each bucket once -- with this request's real latents, on both sides.
            z = _voc_input(dev)
            L = z.shape[-1]
            Lb = _voc_bucket(L)
            if Lb not in voc_done:
                voc_done.add(Lb)
                gold = _fade_out(voc_ref(_voc_pad(z, Lb), voice.speaker_embedding)[:, :, : L * HOP])
                ok_w, pcc_w = comp_pcc(gold, wav, pcc=TARGET_PCC_WAV)
                if verbose:
                    print(f"  {'-> bucket ' + str(Lb):22s} L={L:5d}                 waveform  pcc: {pcc_w}")
                wav_scored.append((f"{label}/voc{Lb}", ok_w, pcc_w))
    finally:
        tts.close()

    worst_label, _, worst_pcc = min(scored, key=lambda r: r[2])
    worst_wav, _, worst_wav_pcc = min(wav_scored, key=lambda r: r[2])
    failed = [label for label, ok, _ in scored + wav_scored if not ok]
    covered = len(buckets) == len(TEXTS)  # one distinct prefill bucket per text, as chosen
    msg = (
        f"{len(scored)} runs over prefill buckets {sorted(buckets)}, deepest cache {deepest}; "
        f"worst latents {worst_label} pcc {worst_pcc}; "
        f"{len(wav_scored)} vocoder buckets {sorted(voc_done)}, worst waveform {worst_wav} "
        f"pcc {worst_wav_pcc}; failed: {failed}"
        + (f"; no codes: {no_codes}" if no_codes else "")
    )
    return not failed and covered and deepest >= MIN_DEPTH and len(voc_done) >= MIN_VOC_BUCKETS, msg


def test_model_teacher_forced_pcc():
    passed, msg = run_teacher_forced_pcc()
    assert passed, f"device latents drifted from the CPU reference for the device's own codes: {msg}"


if __name__ == "__main__":
    import sys

    ok, msg = run_teacher_forced_pcc()
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
