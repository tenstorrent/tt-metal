---
name: mtp-bringup
description: Bring up multi-token prediction (speculative decoding) for TTNN transformer demos like DeepSeek R1, including second-token prediction, cache priming, accept/reject logic, and verification against non-MTP baseline outputs and acceptance-rate thresholds.
---

# MTP Bringup (TTNN DeepSeek)

## Define MTP Behavior
- Predict the next-next token using a lightweight predictor that consumes:
  - Base model hidden state at position t (pre-norm, before RMSNorm)
  - Predicted next token at position t+1
- Verify the prediction by running the base model once on a batched set:
  - Predicted next token (normal decode)
  - MTP-predicted token (speculative decode)

## Integrate Required Signals
- Capture pre-norm hidden state (`hidden_for_mtp`) in both prefill and decode.
- Avoid post-norm hidden states for MTP.
- Use a dedicated MTP module (embedding, norms, decoder block, head).
- Maintain correct concat order in MTP projection. For DeepSeek-R1: `[token_norm, hidden_norm]`.

## Prime MTP Prefill Cache
- Initialize MTP KV cache after base prefill.
- Align inputs to the MTP rule (hidden at t with token at t+1):
  - Keep hidden aligned to cache positions (`hidden_shifted = hidden`).
  - Left-shift token ids to build `token_shifted = [tokens[:, :, 1:], pad_id]`.
  - Keep RoPE unshifted and aligned to the same prefill sequence length.
- Do not prepend pad tokens or add extra timesteps in prefill priming.
- Build shifted tokens on host if TTNN concat/shard causes CCL reduce_scatter failures.
- For multi-device reduce_scatter, the MTP prefill sequence length must be divisible by the ring size. Use the padded `full_seq_len` from `_pad_batch` for MTP prefill, then trim after all_gather.

## Align Decode Positions
- Align MTP decode positions to the token being predicted.
- For the first MTP candidate pass (using `hidden[t]` + `token[t+1]`), use `positions_before+1`.
- For verification of the next-next token, use `positions_before+2`.

## Implement Accept/Reject Logic
- Accept when MTP next-next token equals base verified token.
- Rebatch verification in this order:
  - Batch 0: predicted next token (base)
  - Batch 1: MTP-predicted next-next token
- Advance positions by +2 on accept, +1 on reject.

## Verify End to End
- Run full-model baseline (non-MTP, greedy) and capture output JSON.
- Run full-model MTP (greedy) and require exact output match with baseline.
- Track accept rate (target ~0.8; investigate if <0.5).
- Do not enable MTP for teacher-forcing accuracy; compare baseline vs MTP outputs instead.
- MTP cannot be forced when using `--override-num-layers` (no MTP layer in truncated configs). Use full model for MTP verification.
- Always verify baseline outputs match the known-good reference commit (e.g., `f250fa...`) before judging MTP.

## Watch For Common Failures
- `TT_FATAL reduce_scatter ring_size` in MTP prefill:
  - Avoid mis-sharded inputs.
  - Build shifted tokens on host with identical mesh replication.
  - Ensure the MTP prefill seq_len is padded to the mesh ring size (see above).
- `TT_FATAL ND sharding requires number of chunks`:
  - Avoid creating MTP dummy tensors with a different sharding scheme.
  - Prefer deriving dummy tensors from slices of `hidden_tt` (same sharding) and zeroing via `ttnn.mul(..., 0.0)`.
- Low acceptance:
  - Fix decode position alignment.
  - Fix concat ordering.
  - Ensure pre-norm hidden is used.
  - Ensure MTP prefill uses hidden[t] with token[t+1] and trimmed RoPE.

## Useful Flags and Env
- `--mtp on|off|auto`
- `--compare-output <baseline_json>`
- `--min-mtp-accept-rate <float>`
- `DEEPSEEK_MTP_DEBUG=1` for detailed logs
- Set `DEEPSEEK_V3_MAX_SEQ_LEN=128` for quick testing runs.
- `DEEPSEEK_V3_MAX_SEQ_LEN=<N>` to clamp sequence length

## Likely Files to Touch (DeepSeek v3 Demo)
- `models/demos/deepseek_v3/tt/generator.py`
- `models/demos/deepseek_v3/tt/mtp.py`
- `models/demos/deepseek_v3/tt/model/row_batched_model.py`

## Recent Learnings (2026-02)
- Acceptance can be near-zero even when outputs still look plausible. The highest-leverage first checks are:
  - `positions_tail` for first spec pass should use current decode lengths directly (no `-1` clamp).
  - `positions_for_spec` should be `positions_before+1` for candidate and `positions_before+2` for verification.
- Keep MTP prefill alignment strict and simple:
  - Use full padded prefill length for cache priming.
  - Keep `hidden_shifted` aligned to cache positions and shift tokens to represent `token[t+1]` (pad tail only).
  - RoPE trimming must match the shifted sequence length exactly.
- Mesh/shard shape mismatches show up late and look unrelated:
  - Before MTP concat/projection, force hidden/token sequence dims to match (slice both to `min_len` if needed).
  - When creating helper/dummy tensors, derive from already-sharded tensors to preserve memory layout and avoid ND chunk errors.
- For debug speed, prefer targeted acceptance diagnostics over full traces:
  - Log accepted/rejected lanes with `(position, pred_next, spec_token, pred_after_spec)` tuples.
  - This makes off-by-one and token/hidden misalignment visible within a few steps.

## Additional Learnings For Future Runs (2026-02)
- Use a strict triage order when acceptance regresses:
  - First verify decode position math, then hidden/token alignment, then concat ordering, then mesh/shard shape parity.
  - This avoids spending time on low-probability causes before the common off-by-one failures are ruled out.
- Keep baseline and MTP runs as deterministic as possible during bring-up:
  - Use greedy decode, fixed prompts, and the same max-seq clamp.
  - Compare outputs and acceptance from identical run settings before changing kernels or sharding.
- Treat acceptance and output parity as separate gates:
  - Gate 1: exact output parity vs non-MTP baseline.
  - Gate 2: acceptance threshold (investigate if low, but do not accept output divergence even if acceptance improves).
- Add lightweight stepwise checks before full long-context runs:
  - Validate first accepted token transition and first reject transition in short runs (`max_seq_len=128`).
  - Only then scale to longer contexts to avoid expensive iteration loops.
- When touching prefill/cache code, re-check all three invariants together:
  - MTP prefill uses padded `full_seq_len`.
  - Hidden stays unshifted while tokens are left-shifted with tail pad.
  - RoPE slicing/trim length matches the effective MTP prefill sequence exactly.
- If full demo runs stall right after `Creating model shared states...` with no log growth:
  - Confirm staleness with `stat` on the active log/json before waiting indefinitely.
  - Stop the run and clear stale launcher processes (`pkill -f demo.py`/`prterun`) before retrying.
  - A `reset.sh` run may reinitialize boards successfully but still end with `test_system_health` missing; treat this as a tooling issue, not necessarily a failed hardware reset.
  - Use the most recent completed OFF/ON artifact pair for parity/acceptance gates if reruns are blocked by this stall signature.
- Trace-mode gotchas seen in this bring-up:
  - Default trace region sizing can trigger mailbox-level TT_FATAL on this workload; allow runtime override (for example `DEEPSEEK_TRACE_REGION_SIZE=134217728`).
  - Removing decode warm-up before trace capture can trigger `TT_FATAL: Writes are not supported during trace capture`.
  - A naive warm-up decode on real page tables can mutate decode cache state and break output parity.
  - An isolated warm-up routed to a non-prompt cache row can stabilize capture/hangs, but parity still needs explicit token-level validation.
- Distributed env toggles are not always reliable across launcher ranks:
  - For A/B checks, prefer explicit CLI/runtime wiring over ad-hoc env vars so both ranks use identical settings.
  - In this repo, `--mtp-skip-on-accept on|off|auto` is the deterministic control point for skip-path verification.
- Reference parity requires matching run settings exactly:
  - The stored reference artifact `deepseek_demo_baseline_f250fa_20260216_195652.json` corresponds to the shorter generation configuration (`--max-new-tokens 16`), not 32.
  - If parity fails with otherwise correct code, verify decode length first before deeper debugging.

## Verified Run Snapshot (2026-02-18, patch13)
- Verified artifacts:
  - Reference baseline: `logs/deepseek_demo_baseline_f250fa_20260216_195652.json`
  - MTP off: `logs/deepseek_demo_off_patch13_mnt16_20260218_140532.json`
  - MTP on: `logs/deepseek_demo_on_patch13_mnt16_20260218_140706.json`
- Verification outcomes:
  - Gate 1 (current branch, MTP off vs reference baseline): PASS
  - Gate 2 (MTP on vs MTP off): PASS
  - Gate 3 (acceptance threshold): PASS (`54/66 = 0.818`, above `0.5`)
- Use content-only comparisons for parity checks:
  - Compare `{prompts, generations[index,prompt,text]}` and ignore runtime statistics fields, which are expected to differ.
  - Example check:
    - `jq -S '{prompts, generations: [.generations[] | {index,prompt,text}]}' <json> | diff -u ...`
- Reference integrity guardrail:
  - Record baseline hash before/after bring-up checks:
    - `sha256sum logs/deepseek_demo_baseline_f250fa_20260216_195652.json`
    - Expected for this run: `043822cc45e3a0b66043dab912e9bd59bffb3c74fb07f5beace169c232816f78`
