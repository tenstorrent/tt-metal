# MiniMax-H3: the per-device packed length the tuned tables key on is not the one the pipeline runs

**Status: FIXED in `a07012d7d8a`** (structural; `664578b377e` was the point fix). `_packed_sizes` counts
audio in rows with the gate's 39-token prompt; `packing.py` gained `packed_sequence_length` /
`padded_sequence_length` and the pipeline's own `build_packed_sequence` and `_denoise` padding route
through them; `get_matmul_config` / `get_agmm_config` fall back on equal `M_per_core` after an exact
miss; all M literals re-keyed to 4736 / 9184 / 13664. Confirmed on a live 15 s / 16:9 generation:
`13664 rows/device`, ff1 and ff2 absent from the fallback warnings. Same-host A/B: -69.9 ms/fwd
(-0.58%), matching the isolated-sweep prediction. Both blockings PCC-validated after the fact (ff2 pcc
1.0000000, ff1 0.9999843), since the sweep never checked numerics. Remaining: the "Decision needed"
below is resolved as 39 tokens (matches `CALIBRATED_FOX_PROMPT`; any 1-250 tokens gives the same
15 s bucket). The original analysis follows unchanged.

## TL;DR

Every M-keyed tuning table for H3 at 768P is keyed on **4768 / 9216 / 13632** rows per device
(5 s / 10 s / 15 s). The pipeline actually runs **4736 / 9184 / 13664**. The lookups are exact-key, so
the entries never hit and the tuned blockings are silently replaced by their fallbacks.

The constants come from one formula in `test_performance_minimax_h3.py::_packed_sizes` that
(a) counts audio **latents** where the pipeline packs **latents x 2 channels**, and (b) assumes a
**512-token** prompt where the perf gate runs **39**. Discrepancy (a) is a bug; (b) is an assumption
that has to match the gate either way. Under *either* text budget the existing constants match nothing.

## What the pipeline produces

Logged on every run by `pipeline_minimax_h3.py:1888` (`packed sequence ... rows/device`), and
confirmed as the actual `M` handed to ff2 by the pre-fix `No fused MM/RS config for (M, K, N) = ...`
warnings, which printed it literally:

| 768P duration | frames | latent frames | audio latents | video rows | seq (39-tok prompt) | padded | **rows/device** |
|---|---|---|---|---|---|---|---|
| 5 s  | 124 | 37  | 207 | 37296  | 37749  | 37888  | **4736**  |
| 10 s | 243 | 72  | 405 | 72576  | 73425  | 73472  | **9184**  |
| 15 s | 362 | 107 | 603 | 107856 | 109101 | 109312 | **13664** |

The accounting (`packing.py:263`, `:261`) and padding (`pipeline_minimax_h3.py:1886-1887`):

```python
num_audio_rows  = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS          # channels = 2
sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
alignment  = sp_factor * ttnn.TILE_SIZE                                   # 8 * 32 = 256
padded_len = ((sequence_length + alignment - 1) // alignment) * alignment
# rows/device = padded_len // sp_factor
```

These are the same at `9c96923d1bb` (Aug 13) and HEAD; `packing.py` has had no commits since. There was
no drift — the constants never matched.

## What the tables assume, and where the constants live

| location | keys | consumer |
|---|---|---|
| `attention_minimax_h3.py:84-88` `measured_sdpa_chunk_sizes` | 4768 / 9216 / 13632 | `.get(seq_local)` at `:291`, `seq_local = q_BHNE.shape[2]` |
| `matmul.py:125` `grid_88_configs` | `(13632, 5376, 7168)` ff1 | fused AGMM via `get_agmm_config` |
| `matmul.py:138` `grid_89_configs` | `(13632, 3584, 5376)` ff2 | `RowParallelLinear.forward` -> `get_matmul_config` |
| `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:392-412` | 4768 / 9216 / 13632 | the SDPA perf-table sweep |
| `utils/sweep_mm_block_sizes.py:195-211` | 4768 / 9216 / 13632 | the matmul blocking sweeps |
| `agmm_config.py:9`, `mmrs_config.py:29` | prose only | — |

**Not affected:** `agmm_config.py:49-53` `AGMM_BLOCK_SIZES` is keyed on `(K, N)` deliberately, so ff1's
AGMM default is fine. The `AGMM_BLOCK_SIZES` docstring argues per-duration keying would mean "one
entry per matmul per duration"; that was the right instinct.

## Root cause: `test_performance_minimax_h3.py::_packed_sizes`

Present with this exact shape since `9c96923d1bb` (Aug 13), unchanged since:

```python
NUM_TEXT_TOKENS = 512                                        # :49
num_audio = audio_latent_num_frames(num_frames)              # :61  -- a LATENT count
"seq_len": NUM_TEXT_TOKENS + num_audio + num_video,          # :73  -- audio counted x1
```
then padded exactly as the pipeline does (`:121-123`). `packed_layout` (`tests/.../common.py:222`)
uses its `num_audio` argument as **rows**, so the harness is internally consistent — it simply models
half the audio rows the pipeline has, and a 512-token prompt.

Executed, side by side:

| dur | `_packed_sizes` (512 tok, audio x1) | pipeline-style @512 tok (audio x2) | pipeline @39 tok (audio x2) |
|---|---|---|---|
| 5 s  | **4768**  | 4800  | **4736**  |
| 10 s | **9216**  | 9248  | **9184**  |
| 15 s | **13632** | 13728 | **13664** |

Column 1 reproduces the hardcoded constants exactly. Column 3 reproduces the pipeline logs exactly.
Column 2 shows the audio undercount is decisive on its own: even granting 512 tokens, the pipeline
never lands on 4768/9216/13632.

No single formula built from the pipeline's own helpers produces all three constants; individually
they require mutually inconsistent assumptions (5 s: 179-434 text tokens; 10 s: 87-342; 15 s: **zero
text and no padding**). At 15 s, 13632 is **unreachable at any prompt length**: the media rows alone
(109062) exceed 13632 x 8 = 109056.

## Why it wasn't caught

The Aug 13 commit that hardcoded 4768 also recorded the pipeline's real 5 s length in
`MiniMaxH3.md:277` — "37749 rows for a 39-token prompt ... padded to a multiple of SP x TILE", i.e.
37888 -> **4736** — so the constants and the pipeline disagreed on day one, in the same commit.
The Aug 13 docstring cited `test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table`
with configs `minimax_h3_{5s,10s,15s}_768p` — **none of which existed in the committed tree until
`3a2f7fea259` on 2026-09-16.** The SDPA sweep ran from an uncommitted harness and the results were
transcribed by hand. The pipeline's own `rows/device` log line was the cross-check that would have
caught it, and it was in every log.

## Propagation

- `9c96923d1bb` (Aug 13) — `_packed_sizes` + constants transcribed into `attention_minimax_h3.py`.
- `agmm_config.py`, `mmrs_config.py` restate them as fact in prose.
- `50908410b42` (Sept 16) — `sweep_mm_block_sizes.py` adopts 4768 as "the same anchor the Blackhole
  rows use" and extends to 9216 / 13632.
- `3a2f7fea259` (Sept 16) — the SDPA perf configs reach `tests/` with the same literals.
- `9e97f1541bc` (Sept 17) — the two `matmul.py` entries keyed on 13632.
- `c825d089e31` (Sept 17) — the per-op block breakdown at 15 s was taken at the harness shape.
- `e5c39cbdd47` (Sept 17) — the 15 s SDPA search closes out at `seq_local 13632`; the `q=128` hang it
  documents was also at that shape.

## Impact today

**Dead entries — FIXED.** Both are now keyed on 13664 and verified to resolve:

```
M=13664 (pipeline)  ff1 (8,7,10)  ff2 (8,7,10)   <- tuned, no warnings
M=13632 (harness)   ff1 (8,3,14)  ff2 (8,8,8)    <- now the miss, with warnings
```

The 15665.5 us / -6.3% (ff1) and 6761.3 us / -4.0% (ff2) should now be realized in the pipeline.
`M_per_core` is 54 at both 13632 and 13664, so no re-sweep was needed. The side effect is that the
sweep harness and block perf test — which still report 13632 — now take the fallbacks; fixing that
is the `_packed_sizes` work below.

Verified by executing the real resolvers (see "How to verify"): at M=13632 both hit; at M=13664 both
miss and emit `No known best blocking for (M, K, N) = (13664, ...)`.

**`measured_sdpa_chunk_sizes` never hits at any duration.** Harmless at 10 s and 15 s — the measured
value `(256, 512)` equals the fallback. **Behavioural at 5 s**: the tuned `(320, 384)` never applies
and the pipeline runs the fallback `(256, 512)`; the docstring describes that optimum as sharp (q=320
at 7.81 ms vs q=288 at 10.43 ms), so this is likely a real loss at 5 s. *(seq_local = per-device rows
is inferred from `q_BHNE.shape[2]`, not observed on device.)*

**Sweep measurements were taken at a shape the pipeline never runs.** M is off by 32 (0.23%) and the
audio rows by 603. The blocking optima almost certainly still hold; the absolute numbers in the perf
doc's per-op breakdown describe the harness shape.

**Not Wormhole-specific.** The packing math is architecture-independent, so the same
`measured_sdpa_chunk_sizes` mismatch exists on Blackhole, where the constants were originally
measured. *(Inferred; no Blackhole logs were checked.)*

## Proposed fix (NOT applied)

**1. One source of truth in `packing.py`** (keep it free of `ttnn`):

```python
MINIMAX_H3_TILE = 32

def packed_sequence_length(num_text_tokens, num_audio_latents, num_video_rows, num_condition_rows=0) -> int:
    return num_text_tokens + num_condition_rows + num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS + num_video_rows

def padded_sequence_length(sequence_length, sp_factor, tile=MINIMAX_H3_TILE) -> int:
    alignment = sp_factor * tile
    return ((sequence_length + alignment - 1) // alignment) * alignment
```
and route `build_packed_sequence` (`packing.py:263`) and `_denoise` (`pipeline_minimax_h3.py:1886-1887`)
through them so they cannot diverge.

**2. The bug in `_packed_sizes`:**

```diff
-    num_audio = audio_latent_num_frames(num_frames)
+    num_audio_latents = audio_latent_num_frames(num_frames)
+    num_audio = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS   # ROWS; packed_layout/seq_len/sim_seq_len all count rows
     ...
-        "seq_len": NUM_TEXT_TOKENS + num_audio + num_video,
+        "num_audio_latents": num_audio_latents,
+        "seq_len": packed_sequence_length(NUM_TEXT_TOKENS, num_audio_latents, num_video),
```
```diff
-    alignment = sp_factor * ttnn.TILE_SIZE * SIM
-    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
-    padded_len = padded_len // SIM
+    padded_len = padded_sequence_length(seq_len, sp_factor * SIM) // SIM
```
`sim_seq_len` (`:137`) and the `packed_layout(...)` call (`:140`) already consume `sizes["num_audio"]`
and become correct without edits.

**3. Derive keys instead of hardcoding M.** E.g. a helper in the test `common.py`:

```python
def h3_rows_per_device(duration_s, num_text_tokens, sp_factor=8) -> int:
    s = _packed_sizes(duration_s)
    seq = packed_sequence_length(num_text_tokens, s["num_audio_latents"], s["num_video"])
    return padded_sequence_length(seq, sp_factor) // sp_factor
```
used by `test_ring_joint_sdpa.py:392-412` and `sweep_mm_block_sizes.py:195-211` in place of the
literals. For `matmul.py:125/:138` and `attention_minimax_h3.py:84-88`, either re-key to the gate's
values (13664 etc.; the optimum will not move for 32 rows, so no re-sweep) or — more durably — follow
the pattern `mmrs_config.py` already uses: `register_mmrs_config(m, ...)` is called at the point of use
with the **real** `normed.shape[2]`, so its key can never disagree with the pipeline.

## Decision needed before fixing

`NUM_TEXT_TOKENS = 512` vs the gate's 39-token prompt. With the audio bug fixed, the derived rows are
4800 / 9248 / 13728 at 512 tokens and 4736 / 9184 / 13664 at 39. Whichever budget is chosen, the perf
gate's prompt and the harness constant must agree, or no M-keyed table can ever hit.

## How to verify in 30 s (host only, no device)

```bash
cd /home/jameslee/tt-metal
# 1. what the pipeline ran (from any sweep log)
zcat ~/h3_wormhole_results/sweep_fixed.log.gz | grep -m3 "rows/device"
# 2. do the entries hit at that M?
./python_env/bin/python - <<'PY'
import ttnn
from models.tt_dit.utils import matmul as mm
from models.tt_dit.models.transformers.minimax_h3.agmm_config import agmm_block_size
FULL = ttnn.CoreCoord(8, 9)
for M in (13632, 13664):
    grid, cfg, _ = mm.get_agmm_config(M, 5376, 7168, FULL, 4, 4, default_block_size=agmm_block_size(5376, 7168), fuse_swiglu=True)
    c2 = mm.get_matmul_config(M, 3584, 5376, FULL, None)
    print(M, "ff1", (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size), "ff2", (c2.M_block_size, c2.K_block_size, c2.N_block_size))
PY
# expect: 13632 -> ff1 (8,7,10) ff2 (8,7,10); 13664 -> ff1 (8,3,14) ff2 (8,8,8) plus two "No known best blocking" warnings
```

## Not verified

- What M the block perf harness *reports* when run (only its formula was evaluated).
- SDPA `seq_local` on device (inferred from `q_BHNE.shape[2]`).
- Any Blackhole log.

## Related

- `MiniMaxH3_wormhole_perf.md` — the sweep tables and the open-issues index.
- `MiniMaxH3_wormhole_hang.md` — the MM/RS gate hang; unrelated cause, but the same evidence-preserving
  run recipe (`TT_METAL_OPERATION_TIMEOUT_SECONDS` + `tt-triage`) would have captured the `q=128`
  SDPA hang in `e5c39cbdd47` on its first occurrence instead of wedging the board twice.
