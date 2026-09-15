# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end RTF -- the bring-up's headline perf gate: RTF < 0.5, then < 0.2.

Real-time factor is compute seconds per second of audio produced. The three
stages contribute very differently, and the split is the whole story:

* the **LLM runs once per token**, and a second of speech is 50 tokens, so its
  contribution is `50 / tok_s` -- it is the only stage whose cost scales with
  the length of the output rather than being amortised over it;
* the **flow decoder** runs ten Euler steps over the whole utterance at once;
* the **vocoder** runs once.

So the flow and the vocoder get cheaper per second as utterances get longer,
while the LLM does not. Reporting a single RTF without that breakdown would hide
which stage to optimise.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.cosyvoice.tests.perf.gates import enforce, report
from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden
from models.demos.cosyvoice.tt.weights import default_weights_path

HIFT_WEIGHTS = default_weights_path()
FLOW_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "flow_")
LLM_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "llm_")
SAMPLE_RATE = 22050

needs_l1_small = pytest.mark.parametrize(
    # 384 MB, not the usual 64: the in-place KV cache captures 65 traces where the
    # moving one captures a single trace. The requirement is not a tidy per-trace
    # figure -- offered 64 MB it asked for 68.6, offered 128 it asked for 134.3, so
    # the allocator is filling whatever it is given before it reports a shortfall.
    # 384 MB is simply a size this class has been observed to capture in.
    "device_params",
    [{"l1_small_size": 131072, "trace_region_size": 402653184}],
    indirect=True,
)
needs_all = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (HIFT_WEIGHTS, FLOW_WEIGHTS, LLM_WEIGHTS)),
    reason="export hift, flow and llm weights first",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "e2e.npz")), reason="generate goldens first"
)


@needs_all
@needs_golden
@needs_l1_small
def test_device_end_to_end_rtf(device):
    """Time each stage at the captured utterance's real shapes and combine.

    The LLM is timed per decode step and scaled to the token count rather than
    generated for real: sampling would give a different number of tokens each run
    and make the figure irreproducible, while the per-step cost -- which is what
    actually determines RTF -- is identical either way.
    """
    import ttnn
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")
    want = as_torch(load_golden("e2e")["waveform"])

    audio_seconds = want.shape[1] / SAMPLE_RATE
    tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
    token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
    n_generated = tokens.shape[1] - token_len1
    mel_len1, mel_len2 = int(lr_g["call0.in_mel_len1"]), int(lr_g["call0.in_mel_len2"])

    def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    # ---------------------------------------------------------------- LLM
    llm_bag = WeightBag.load(LLM_WEIGHTS)
    ar_meta = llm_bag.meta["ar_decoder"]
    dec = TtARDecoder(device, llm_bag.sub("llm"), ar_meta)
    d = ar_meta["input_size"]
    prefix_len = 209
    max_len = ((prefix_len + n_generated + 1 + 127) // 128) * 128
    torch.manual_seed(0)
    prefix, step_in = torch.randn(1, prefix_len, d) * 0.1, torch.randn(1, 1, d) * 0.1

    def prefill():
        c = dec.empty_cache(max_len, prefix_len)
        ys, c = dec.forward_chunk_fixed(
            dev(prefix), c, max_len, prefix_len, dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True))
        )
        ttnn.deallocate(ys)
        return c

    # Traced decode -- the path `generate()` takes. The KV-cache mode is read from
    # the same environment variable `generate()` reads, so this measures whatever the
    # model would actually run rather than a fixed choice that can drift away from it.
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TracedDecodeStepInPlace, kv_inplace_default

    _kv_env = os.environ.get("COSYVOICE_KV_INPLACE")
    use_inplace = (_kv_env == "1") if _kv_env is not None else kv_inplace_default(device)
    kls = TracedDecodeStepInPlace if use_inplace else TracedDecodeStep
    caches = prefill()
    traced = kls(dec, max_len).capture()
    traced.seed(caches)
    TtARDecoder.free_caches(caches)
    for i in range(2):
        traced.step(step_in, prefix_len + 1 + i)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for i in range(16):
        traced.step(step_in, prefix_len + 3 + i)
        ttnn.synchronize_device(device)
    llm_step_ms = (time.perf_counter() - t0) / 16 * 1e3
    traced.release()
    llm_total_s = llm_step_ms * n_generated / 1e3

    # --------------------------------------------------------------- flow
    flow_bag = WeightBag.load(FLOW_WEIGHTS)
    flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)

    def flow_args():
        """Rebuilt per call: `solve_euler` consumes `z` on its first step, so the
        arguments are not reusable across invocations."""
        return (
            dev(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            token_len1,
            mel_len1,
            mel_len2,
            dev(as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()),
            dev(as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)),
            dev(as_torch(cfm_g["call0.rng_z"]).permute(0, 2, 1).contiguous()),
        )

    ttnn.deallocate(flow.inference(*flow_args()))  # warm-up
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    mel = flow.inference(*flow_args())
    ttnn.synchronize_device(device)
    flow_s = time.perf_counter() - t0

    # ------------------------------------------------------------ vocoder
    hift = TtHiFTGenerator(device, WeightBag.load(HIFT_WEIGHTS))
    sine_g = load_golden("hift.sinegen")
    phase = dev(as_torch(sine_g["call0.in_phase_vec"]), dtype=ttnn.float32)
    noise = dev(as_torch(sine_g["call0.out_noise"]).permute(0, 2, 1).contiguous(), dtype=ttnn.float32)
    wav, _, src = hift.inference(mel, mel_len2, phase_vec=phase, sine_noise=noise)  # warm-up
    ttnn.deallocate(src)
    ttnn.deallocate(wav)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    wav, _, src = hift.inference(mel, mel_len2, phase_vec=phase, sine_noise=noise)
    ttnn.synchronize_device(device)
    hift_s = time.perf_counter() - t0
    ttnn.deallocate(wav)
    ttnn.deallocate(src)
    ttnn.deallocate(mel)

    total_s = llm_total_s + flow_s + hift_s
    rtf = total_s / audio_seconds
    print(f"\n  utterance: {n_generated} generated tokens -> {audio_seconds:.2f} s of audio")
    print(
        f"  LLM      {llm_step_ms:7.2f} ms/token x {n_generated:4d} = {llm_total_s:6.3f} s"
        f"   RTF {llm_total_s/audio_seconds:5.3f}  ({1e3/llm_step_ms:.1f} tok/s)"
    )
    print(f"  flow     10 Euler steps               = {flow_s:6.3f} s   RTF {flow_s/audio_seconds:5.3f}")
    print(f"  vocoder  mel -> {want.shape[1]} samples        = {hift_s:6.3f} s   RTF {hift_s/audio_seconds:5.3f}")
    print(f"  TOTAL                                 = {total_s:6.3f} s   RTF {rtf:5.3f}")
    print(f"  LLM share of total: {100*llm_total_s/total_s:.1f}%")

    # The gates, enforced. Every threshold is asserted -- a met one against the
    # requirement itself, a missed one against its recorded band, both bounds. See
    # `gates.py` for why a missed target is not an `xfail`.
    tok_s = 1e3 / llm_step_ms
    report(
        [
            enforce("tok_s", tok_s, device),
            enforce("tok_s_stretch", tok_s, device),
            enforce("rtf", rtf, device, extra=f"{n_generated} tokens, {audio_seconds:.2f} s audio"),
            enforce("rtf_stretch", rtf, device),
        ],
        "bounty gates -- end-to-end",
    )
