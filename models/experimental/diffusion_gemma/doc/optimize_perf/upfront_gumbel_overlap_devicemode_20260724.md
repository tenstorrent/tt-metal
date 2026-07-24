# Up-front denoise: host-Gumbel overlap + on-device Gumbel (2026-07-24)

Two optimizations to the up-front traced denoise path (`tt/traced_denoise.py`
`UpfrontTracedDenoiseController`), motivated by the finding that the path is **host-bound,
not device-bound**: in the served `DG_VLLM_GUMBEL_MODE=host` contract every replay step
regenerates a full-vocab `torch.rand((1,256,262144))` Gumbel (256 MiB, ~313 ms host CPU) and
replicates it to all 4 devices (~1 GiB H2D DMA) — with the device idle for the host RNG and a
redundant per-step `synchronize_device` foreclosing any overlap. Wan (the reference diffusion
model) keeps its whole denoise loop on-device with no per-step host RNG and no per-step readback;
these changes move DiffusionGemma toward that shape.

## Opt B — host-Gumbel prefetch + drop the per-step sync (DEFAULT ON, byte-identical)

- `tt/generate.py` `make_seeded_host_gumbel_noise_fn`: the next step's host Gumbel `torch.rand`
  is computed on a 1-thread worker while the current step's device trace runs; only the
  `from_torch` upload stays on the main thread (no concurrent device access). Gated by
  `DG_HOST_GUMBEL_PREFETCH` (default 1; `0` = exact serial baseline). Byte-identical because the
  Gumbel is per-`(block,step)` privately seeded — only *when* it is generated changes, never the
  value. The shared-generator renoise-token stream is deliberately **not** touched.
- `tt/traced_denoise.py`: removed the per-step `ttnn.synchronize_device` (the following
  `read_halt_scalars` to_torch is CQ0-ordered + blocking, so ordering is preserved).

Evidence (QB2 `bh-qbge-06`, P150x4, mesh (1,4) TP=4):

- Device-free: prefetch ON vs OFF Gumbel sequences byte-identical + match a standalone
  deterministic draw.
- Reduced-layer device test `test_device_upfront_matches_eager_tokens_realized_k_and_halt`
  (2 layers): upfront-traced committed tokens `torch.equal` to eager, realized-K + halt match.
- Full 30L A/B (p_max=1024, prompt "Explain why the sky is blue", 2 repeats): per-block halt
  step counts **identical** prefetch on/off (22/17/10) — decisions unchanged.
- Throughput: decode-block mean **9.98 s (PF1) vs 11.48 s (PF0)** → **1.15x**; denoise-only
  per-step ~658 ms vs ~771 ms (~1.17x).
- GPQA host smoke @3072 (samples 0,1): doc-0 correct `\boxed{C}` (exact_match=1), coherent.

Byte-identical → no decision re-gate required. This is the shipped default behavior change.

## Opt A — on-device Gumbel (`DG_VLLM_GUMBEL_MODE=device`, NON-default, needs re-gate)

- `tt/generator_vllm.py`: up-front validator now accepts `gumbel_mode in {host, device}`
  (chunked/argmax still rejected loudly — not materialized full tensors).
- Server maps `device` → `make_seeded_gumbel_noise_fn` → `sample_gumbel_noise_with_permuted_vocab`
  (already controller-compatible; RNG runs outside the trace ⇒ trace-safe). Launcher default
  stays `host`.

### Bug fixed en route: 8 GiB OOM in `sample_gumbel_noise_with_permuted_vocab`

The permuted path built `ttnn.rand([vocab, 1, canvas, 1])`; `TILE_LAYOUT` pads the trailing
size-1 axis to a full 32-tile, inflating the `[1,1,256,262144]` buffer **256 MiB → 8 GiB**
(`TT_FATAL bank_manager.cpp:462`, `allocate 8589934592 B`). Fixed by generating a 2-D
`[vocab, inner]` rand (vocab outermost so still not the correlated innermost axis; all non-vocab
dims collapsed into one tile-aligned inner axis), then permute vocab→innermost + reshape. Buffer
back to 256 MiB. Distribution property (vocab-outermost draw) preserved; exact values change.
Unit test `test_permuted_vocab_gumbel_noise_deallocates_pre_permute_tensor` updated (11/11 pass).

### Results

- Device path runs through the full vLLM server (capture + 7-block serve, all early-halting) and
  is **deterministic** (committed-token sha256 stable across repeats).
- Throughput (p_max=1024, same prompt): denoise-only per-step **~428 ms** vs host-serial ~771 ms
  = **~1.8x**, and ~1.54x on top of Opt B. It removes the ~1 GiB/step H2D DMA that Opt B cannot
  hide — this is the larger lever.
- **Quality re-gate — NOT a clean pass at n=2.** GPQA @3072, samples 0/1, same-samples A/B:

  | doc | target | host@3072 | device@3072 |
  |----|--------|-----------|-------------|
  | 0  | C      | em=1, `\boxed{C}` | em=0, correct reasoning → "Answer: (C)" but **no `\boxed{}`** |
  | 1  | A      | em=0, `\boxed{C}` (hard miss) | em=0, `\boxed{C}` (same miss) |
  | **acc** |    | **0.5**   | **0.0**     |

  Device-mode Gumbel perturbed doc-0 generation enough to drop the `\boxed{}` wrapper (reasoning
  still reaches the correct C), costing the one point host earned. Reasoning quality is preserved;
  exact_match regressed on the single sample host got right.

### Verdict / decision (2026-07-24)

Opt A is a real ~1.8x denoise win and is deterministic. The n=2 re-gate was inconclusive-to-
negative (a format-sensitive doc-0 miss: correct reasoning, no `\boxed{}`). **On owner decision the
served default is flipped to `device`** (launcher `DG_VLLM_GUMBEL_MODE:-device`), accepting the n=2
caveat because the two changes ship as **separate commits** so device mode can be reverted
independently (revert the Opt A commit → back to `host` + prefetch) if a problem surfaces. Opt B
(byte-identical) ships as the always-on default. **A sub-40 GPQA host-vs-device @3072 re-gate
remains the recommended follow-up** to confirm answer-parity at scale.

Commands: `run_upfront_gpqa.sh smoke` with `DG_VLLM_GUMBEL_MODE={host,device}` and
`MAX_GEN_TOKS=3072`; throughput via `bench_gumbel_mode.py` / host A/B via `bench_upfront_prefetch.py`.
All runs: 4× Blackhole p300c, no Tracy/watcher.
