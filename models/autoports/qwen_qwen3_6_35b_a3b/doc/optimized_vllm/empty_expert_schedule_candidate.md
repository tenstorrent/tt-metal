# Archived empty-expert schedule candidate

Prepared source-only. `empty_expert_schedule_candidate.patch` changes12 files (172 insertions,6 deletions): the native public/primitive declarations and forwarding, nanobind default, hashed operation attributes, validation, factory defines, three kernels, and model/decoder opt-in. No active production file was edited. Native C++ and device compilation have **not** been run.

## Contract

Native opt-in: `ttnn.experimental.deepseek_prefill.moe_fused_swiglu(..., compact_empty_experts=True)`. DefaultFalse preserves existing callers. ExplicitTrue requires at most64 local experts, no fused bias and `stochastic_rounding=False`; violations fail validation. Both attribute-name/value tuples include the flag, and all three kernel descriptors receive the same compile define. The public and primitive signatures append a defaultFalse argument, preserving source compatibility for existing callers after rebuilding.

Model opt-in: `QwenModel(..., compact_empty_moe_experts=True)`, defaultFalse. The flag propagates to each decoder and is forwarded only for deterministic grouped decode (`stochastic_rounding=False`). Current prefill uses stochastic grouped routing and passes no new keyword. There are no plugin, adapter, scheduler, tensor refresh or sampling changes.

The reader’s opt-in local `expert_plan` lambda preserves the existing complete ID/band/count-clamp/region validation and supplies both the bitmap scan and active body. Default compilation retains the original inline predicate text. A header uses two existing mailbox payload words with magic `MAILBOX_MAGIC + EXPERTS_PER_CHIP`; each consumer snapshots it before popping. UNPACK broadcasts the two bitmap words once to MATH and PACK. Every kernel then visits original local IDs in ascending order, skipping exactly the same zero-block entries before their old mailbox/body. Active experts retain original weight addresses, mailbox magic, per-block counters, arithmetic and padding. Final reader ACK drain, writer pending-output drain and all existing barriers remain untouched.

See `empty_expert_schedule_design.md` for the protocol argument and remaining risks. The extra header does not make a measured performance promise. Dense masks can pay overhead while skipping no experts.

## New focused probe

`tests/run_empty_expert_schedule_probe.py` directly compares native default and opt-in programs on1x4, an8x8 worker grid,64 distinct local expert matrices, BFP8 weights by default (`--weight-dtype bfp8`, matching the selected `expert_ep` policy), BF16 row-major input and BF16 tiled output. Default arithmetic dimensions are256x256 to bound load time; `--embedding 2048 --hidden 512` can exercise Qwen-sized geometry if confirmed against the selected checkpoint. It allocates fixed4096-row output buffers and places a32-row sentinel guard after every expert region.

Nine routing cases run eager and through two retained traces: all-empty, expert63, experts0/31/32/63, mixed289/97/33/1 counts, all64 experts, changed global-ID mappings, unchanged mapping/mask, active→empty and empty→late63. Ranks1/2 have shifted local activity and rank3 is empty in sparse cases; dense activates every rank. Full BF16 output bits must match on all four chips, including untouched guards and inactive regions. Padding within a written tile-row is compared with the reference, not required to preserve its initial sentinel.

The probe verifies that omitted and explicitFalse share the default cache entry, True creates a separate program, repeated calls do not grow the cache, and fresh output addresses bind correctly. Both kernels, metadata updates, output resets and host read paths warm before capture. All input/weight/output tensors predate capture; no captured allocation is acknowledged and no broad exemption is used. Tracker1/skip0 is required. Replays are nonblocking; host comparisons follow the async boundary. Trace release and mesh close run in `finally`; cleanup failures fail the artifact. A fresh output artifact path is mandatory.

After the hardware owner applies the archive and rebuilds/reloads the native TTNN library/binding, run under its serialized external timeout:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_empty_expert_schedule_probe --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/empty_expert_schedule/native.json
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_empty_expert_schedule_probe --min-active-tokens 33 --max-active-tokens 128 --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/empty_expert_schedule/banded.json
```

The banded run verifies that scheduling derives from the accepted runtime band rather than raw nonzero counts. Neither microprobe gives a serving latency or model-accuracy result. After these gates, add the model flag to candidate construction only in the existing reduced/full40 harnesses, retain the original control, then repeat mixed-shape trace reuse and matched native/serving measurements. Test harnesses were not modified to enable the model flag in this task.

## Completed CPU checks

- Python compilation passed for the new probe and two temporary model source copies.
- New-probe precommit hooks passed; temporary C++ sources were formatted using the checkout’s clang-format policy, and temporary Python sources with Black23/120 columns.
- CPU mode validated all nine metadata cases for both the full and33–128 bands. Artifacts are `/tmp/qwen-empty-expert-__y25f50/cpu_metadata.json` and `cpu_metadata_banded.json`; both explicitly record `hardware_verified=false`, with zero device comparisons.
- `git apply --check` passed against the then-current production sources.
- Removing the disabled `COMPACT_EMPTY_EXPERTS` branches from each temporary kernel reproduces the corresponding original source ignoring whitespace. No default kernel logic or barrier is changed.

Temporary copies are under `/tmp/qwen-empty-expert-__y25f50/{base,new}`. Native/primitive signature expansion changes the compiled C++ ABI, so rebuild dependent TTNN targets and load a fresh process; do not mix an old binary with the new Python keyword. Device kernels will compile cold for the new cached signature. No build duration was measured. The exact configured rebuild/install sequence and a second mailbox-lifetime audit are recorded in `empty_expert_schedule_recheck.md`.

Authorization boundary: this is a shared native TTNN op addition with a default-preserving opt-in, explicitly delegated by the parent under the user’s native-change scope. It introduces no plugin or vLLM change and requests no additional authorization. Compilation, hardware protocol correctness, full-model exactness and performance remain unverified.
