# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3ASRGenerator through the vLLM generator API: paged KV, batch > 1, audio prefill.

  test_generator_teacher_forced_pcc   prefill + decode logits vs the CPU Qwen3 decoder on
                                      the same merged embeddings, feeding the reference
                                      tokens at every step
  test_device_splice_matches_host     on-device audio splice equals the host splice exactly
  test_generator_batch_matches_single greedy on-device decode of 8 requests in one batch
                                      equals each request decoded alone
  test_generator_session_continuation a streaming session (vLLM /v1/realtime): each 5 s
                                      segment prefills on top of the session's KV, builds
                                      no program after warmup, and transcribes as well as
                                      re-prefilling the whole session (WER vs <name>.txt)

Needs the same data as test_interleaved_lengths.py (QWEN3ASR_INTERLEAVE_DIR with
<name>.wav files) plus the Qwen3-ASR snapshot and extracted text decoder (see
demo/transcribe.py). Run on one N150:
  QWEN3ASR_INTERLEAVE_DIR=/data/interleave \
  pytest models/demos/audio/qwen3_asr/tests/test_generator.py
"""
import json
import os
import re

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.common.sampling.generator import SamplingParams
from models.demos.audio.qwen3_asr.demo import transcribe as T
from models.demos.audio.qwen3_asr.tt.generator import AUDIO_TOKEN_ID, Qwen3ASRGenerator
from tests.ttnn.utils_for_testing import comp_pcc

DEVICE_PARAMS = {"l1_small_size": 32768, "trace_region_size": 200000000}
MAX_BATCH = 8
MAX_SEQ_LEN = 2048
BLOCK_SIZE = 64
STEPS = 24
# The N150 decode settings (LoFi MLP, BFP8 KV / QKV / WO) lower single steps to ~0.89;
# without them every step is >= 0.98 on the same path.
PCC_STEP_MIN = 0.85
PCC_STEP_MEAN = 0.97
MIN_TOP1_MATCH = 0.9


def _clip_paths():
    root = os.environ.get("QWEN3ASR_INTERLEAVE_DIR")
    if not root or not os.path.isdir(root):
        pytest.skip("set QWEN3ASR_INTERLEAVE_DIR to a dir of <name>.wav files")
    return [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".wav")]


def _requests(paths):
    """(input_ids, mel) per clip, built as the Qwen3-ASR processor builds them."""
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    snap = T.find_snap()
    tok = AutoTokenizer.from_pretrained(T.find_text_decoder(snap))
    with open(os.path.join(snap, "chat_template.json")) as fh:
        chat_template = json.load(fh)["chat_template"]
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    return [T.build_inputs(T.load_wav(p), fe, tok, chat_template) for p in paths]


def _build(device):
    snap = T.find_snap()
    return Qwen3ASRGenerator.build(
        device, MAX_BATCH, MAX_SEQ_LEN, snap_dir=snap, text_decoder_dir=T.find_text_decoder(snap)
    )


def _paged(gen):
    """KV cache and a shuffled page table with MAX_SEQ_LEN of blocks per slot, with prefill
    warmed up as vLLM does before serving."""
    args = gen.model_args[0]
    blocks_per_slot = MAX_SEQ_LEN // BLOCK_SIZE
    num_blocks = MAX_BATCH * blocks_per_slot
    kv_cache = gen.allocate_kv_cache(
        (num_blocks, args.n_kv_heads, BLOCK_SIZE, args.head_dim), torch.bfloat16, args.n_layers
    )
    page_table = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(0)).reshape(MAX_BATCH, -1)
    gen.warmup_model_prefill(kv_cache, enable_trace=False, can_sample_on_device=True)
    return kv_cache, page_table.to(torch.int32)


def _prefill(gen, kv_cache, page_table, reqs, slots, sampling_params=None, start_pos=None):
    """reqs: (input_ids, mel or list of mels) per user."""
    lens = [int(ids.shape[0]) for ids, _ in reqs]
    tokens = torch.zeros(len(reqs), max(lens), dtype=torch.long)
    for i, (ids, _) in enumerate(reqs):
        tokens[i, : lens[i]] = ids
    return gen.prefill_forward(
        tokens,
        page_table=page_table[slots],
        kv_cache=kv_cache,
        prompt_lens=torch.tensor(lens),
        empty_slots=slots,
        sampling_params=sampling_params,
        start_pos=None if start_pos is None else torch.tensor(start_pos),
        input_audio_features=[mel for _, mel in reqs],
        audio_feature_lengths=[None] * len(reqs),
    )


def _decode(gen, kv_cache, page_table, tokens, positions, sampling_params=None):
    """One decode step over all MAX_BATCH rows. Returns logits (B, vocab) or tokens (B,)."""
    out = gen.decode_forward(
        tokens.reshape(MAX_BATCH, 1),
        positions,
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=sampling_params,
    )
    if isinstance(out, tuple):
        out = out[0]
    if sampling_params is not None:
        return out.reshape(-1)[:MAX_BATCH].long()
    return out.reshape(MAX_BATCH, -1)[:, : gen.model_args[0].vocab_size].float()


def _host_embeds(gen, ids, mel):
    """Prompt embeddings spliced on host: token table rows with the encoder output over the
    audio placeholders."""
    with safe_open(os.path.join(T.find_text_decoder(T.find_snap()), "model.safetensors"), "pt") as h:
        x = h.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)[ids]
    ((audio, n),) = gen.encode_audio(mel)
    x[ids == AUDIO_TOKEN_ID] = ttnn.to_torch(audio).reshape(-1, x.shape[-1])[:n].to(torch.bfloat16)
    ttnn.deallocate(audio)
    return x


def _reference_logits(ckpt, embeds, steps):
    """CPU Qwen3 decoder from merged embeddings, greedy: logits at the prompt's last
    position and after each greedy token, plus those tokens."""
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    logits, toks = [], []
    with torch.no_grad():
        out = ref(inputs_embeds=embeds.float().unsqueeze(0), use_cache=True)
        for _ in range(steps + 1):
            step = out.logits[0, -1].float()
            logits.append(step)
            toks.append(int(step.argmax()))
            out = ref(input_ids=torch.tensor([[toks[-1]]]), past_key_values=out.past_key_values, use_cache=True)
    return logits, toks


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_generator_teacher_forced_pcc(device):
    reqs = _requests(_clip_paths()[:1])
    gen = _build(device)
    kv_cache, page_table = _paged(gen)
    ids, mel = reqs[0]
    embeds = _host_embeds(gen, ids, mel)
    ref_logits, ref_toks = _reference_logits(T.find_text_decoder(T.find_snap()), embeds, STEPS)

    tt_logits = [_prefill(gen, kv_cache, page_table, reqs, [0])[0].reshape(-1)[: gen.model_args[0].vocab_size].float()]
    tokens = torch.zeros(MAX_BATCH, dtype=torch.long)
    positions = torch.zeros(MAX_BATCH, dtype=torch.long)
    for k in range(STEPS):
        tokens[0], positions[0] = ref_toks[k], ids.shape[0] + k
        tt_logits.append(_decode(gen, kv_cache, page_table, tokens, positions)[0])

    pccs = [comp_pcc(r, t, PCC_STEP_MIN)[1] for r, t in zip(ref_logits, tt_logits)]
    top1 = sum(int(t.argmax()) == r for t, r in zip(tt_logits, ref_toks)) / len(ref_toks)
    print(f"teacher-forced PCC min {min(pccs):.4f} mean {sum(pccs) / len(pccs):.4f}; top-1 match {top1:.2f}")
    assert min(pccs) >= PCC_STEP_MIN, f"per-step PCC {pccs}"
    assert sum(pccs) / len(pccs) >= PCC_STEP_MEAN, f"per-step PCC {pccs}"
    assert top1 >= MIN_TOP1_MATCH, f"top-1 match {top1:.2f}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_device_splice_matches_host(device):
    """The on-device audio splice gives exactly the host-spliced prompt embeddings."""
    gen = _build(device)
    model = gen.model[0]
    for ids, mel in _requests(_clip_paths()):
        expected = _host_embeds(gen, ids, mel)
        encoded = gen.encode_audio(mel)
        runs = gen.audio_runs(ids)
        model.set_prefill_audio([[(off, n, emb) for (off, n), (emb, _) in zip(runs, encoded)]])
        S = ((ids.shape[0] + 127) // 128) * 128
        tokens = torch.nn.functional.pad(ids, (0, S - ids.shape[0])).reshape(1, -1)
        try:
            x = model.prepare_inputs_prefill(tokens, page_table=None)[0]
        finally:
            model.set_prefill_audio(None)
        got = ttnn.to_torch(x).reshape(-1, expected.shape[-1])[: ids.shape[0]]
        assert torch.equal(got, expected), f"max diff {(got.float() - expected.float()).abs().max()}"
        for emb, _ in encoded:
            ttnn.deallocate(emb)


def _greedy_params(n):
    return SamplingParams(temperature=[0.0] * n, top_k=[1] * n, top_p=[1.0] * n)


def _greedy(gen, kv_cache, page_table, reqs, slots, steps):
    """Greedy on-device decode of reqs in the given slots; other rows idle at position 0."""
    first = _prefill(gen, kv_cache, page_table, reqs, slots, sampling_params=_greedy_params(len(reqs)))
    if isinstance(first, tuple):
        first = first[0].reshape(len(reqs), -1)[:, 0]
    else:
        first = first.reshape(len(reqs), -1)[:, : gen.model_args[0].vocab_size].argmax(-1)
    out = {s: [int(first[i])] for i, s in enumerate(slots)}
    tokens = torch.zeros(MAX_BATCH, dtype=torch.long)
    positions = torch.zeros(MAX_BATCH, dtype=torch.long)
    for i, s in enumerate(slots):
        positions[s] = reqs[i][0].shape[0]
    row_greedy = _greedy_params(MAX_BATCH)
    for _ in range(steps):
        for s in slots:
            tokens[s] = out[s][-1]
        nxt = _decode(gen, kv_cache, page_table, tokens, positions, sampling_params=row_greedy)
        for s in slots:
            out[s].append(int(nxt[s]))
            positions[s] += 1
    return [out[s] for s in slots]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_generator_batch_matches_single(device):
    paths = _clip_paths()
    reqs = _requests([paths[i % len(paths)] for i in range(MAX_BATCH)])
    gen = _build(device)
    kv_cache, page_table = _paged(gen)
    # Programs first built after a trace is captured can be overwritten by its replays, so
    # audio prefill must build none after warmup, whatever the clip length.
    _greedy(gen, kv_cache, page_table, reqs[:1], [0], 1)
    cached = device.num_program_cache_entries()
    for r in reqs:
        _prefill(gen, kv_cache, page_table, [r], [0], sampling_params=_greedy_params(1))
    assert device.num_program_cache_entries() == cached, "audio prefill built new programs after warmup"

    single = [_greedy(gen, kv_cache, page_table, [r], [0], STEPS)[0] for r in reqs]
    # Reversed slots, so each request also moves to a different KV slot and page set.
    slots = list(range(MAX_BATCH))[::-1]
    batched = _greedy(gen, kv_cache, page_table, reqs, slots, STEPS)

    for i, (a, b) in enumerate(zip(single, batched)):
        assert a == b, f"request {i}: batch-1 {a} != batch-{MAX_BATCH} {b}"


SEGMENT_S = 5
SESSION_STEPS = 32
SESSION_MAX_WER = 0.1
SESSION_WER_MARGIN = 0.02
IM_END = 151645


def _segments(path):
    """A clip cut into SEGMENT_S pieces, each as vLLM's Qwen3-ASR realtime prompt (no
    system turn): (input_ids, mel)."""
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    snap = T.find_snap()
    tok = AutoTokenizer.from_pretrained(T.find_text_decoder(snap))
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    scaffold = tok("<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n")[
        "input_ids"
    ]
    cut = scaffold.index(AUDIO_TOKEN_ID)
    wav = T.load_wav(path)
    step = SEGMENT_S * T.SR
    out = []
    for lo in range(0, len(wav), step):
        feats = fe(wav[lo : lo + step], sampling_rate=T.SR, return_attention_mask=True, return_tensors="pt")
        flen = int(feats["attention_mask"].sum())
        mel = feats["input_features"][0][:, :flen].float()
        n = T.feat_out_len(flen)
        ids = torch.tensor(scaffold[:cut] + [AUDIO_TOKEN_ID] * n + scaffold[cut + 1 :], dtype=torch.long)
        out.append((ids, mel))
    return out


def _session(gen, kv_cache, page_table, segments, slot, continue_kv):
    """Greedy outputs per segment of a streaming session in one slot. With continue_kv each
    segment prefills only its own prompt on top of the session's KV, as vLLM schedules a
    streaming session; without it each segment re-prefills the whole session from 0."""
    seq = torch.zeros(0, dtype=torch.long)
    mels = []
    computed = 0
    outputs = []
    tokens = torch.zeros(MAX_BATCH, dtype=torch.long)
    positions = torch.zeros(MAX_BATCH, dtype=torch.long)
    row_greedy = _greedy_params(MAX_BATCH)
    for ids, mel in segments:
        seq = torch.cat([seq, ids])
        mels.append(mel)
        first = _prefill(
            gen,
            kv_cache,
            page_table,
            [(seq, list(mels))],
            [slot],
            sampling_params=_greedy_params(1),
            start_pos=[computed if continue_kv else 0],
        )
        if isinstance(first, tuple):
            first = first[0]
        toks = [int(first.reshape(-1)[0])]
        computed = seq.shape[0]
        while toks[-1] != IM_END and len(toks) <= SESSION_STEPS:
            tokens[slot], positions[slot] = toks[-1], computed
            toks.append(int(_decode(gen, kv_cache, page_table, tokens, positions, sampling_params=row_greedy)[slot]))
            computed += 1
        outputs.append(toks)
        # vLLM keeps every computed output token as session context; the last sampled one
        # never went through the model.
        seq = torch.cat([seq, torch.tensor(toks[:-1], dtype=torch.long)])
    return outputs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_generator_session_continuation(device):
    paths = sorted(_clip_paths(), key=os.path.getsize)
    gen = _build(device)
    kv_cache, page_table = _paged(gen)
    # Captures the decode trace, so the session below runs while it exists.
    _session(gen, kv_cache, page_table, _segments(paths[-1])[:2], 0, continue_kv=True)
    cached = device.num_program_cache_entries()
    segments = _segments(paths[-1]) + _segments(paths[0])
    session = _session(gen, kv_cache, page_table, segments, 1, continue_kv=True)
    assert device.num_program_cache_entries() == cached, "session continuation built new programs after warmup"
    full = _session(gen, kv_cache, page_table, segments, 2, continue_kv=False)
    tok = gen.tokenizer
    reference = _words(" ".join(open(p[: -len(".wav")] + ".txt").read() for p in (paths[-1], paths[0])))
    wer = {}
    for name, outputs in (("continued", session), ("full prefill", full)):
        text = " ".join(T.parse_asr(tok.decode([t for t in toks if t != IM_END]))[1] for toks in outputs)
        wer[name] = _word_errors(_words(text), reference) / len(reference)
        print(f"{name}: WER {wer[name]:.3f} {text!r}")
    # Continued and full prefill attend through different SDPA paths over a BFP8 KV cache,
    # so single words at segment cuts differ; neither path may be worse overall.
    assert wer["continued"] <= wer["full prefill"] + SESSION_WER_MARGIN, wer
    assert wer["continued"] <= SESSION_MAX_WER, wer


def _words(text):
    words = re.sub(r"[^A-Z0-9' ]", " ", text.upper()).split()
    return ["MISTER" if w == "MR" else w for w in words]


def _word_errors(hyp, ref):
    d = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (h != r))
    return d[-1]
