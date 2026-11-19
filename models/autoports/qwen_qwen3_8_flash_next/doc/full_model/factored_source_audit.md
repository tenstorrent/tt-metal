# Factored decode preparatory source audit

CPU/source review, 2026-11-14. This is not a stage-review verdict, full-model acceptance, or a new device measurement. No production source was changed and no device job was launched for this audit. The parent was running TT66 independently.

No concrete state-ordering, dtype/fidelity, indexed-copy, or retained-trace ownership defect was found in the revision below. The remaining integration evidence is described explicitly below; reduced diagnostic equivalence does not replace it.

## Revision

Paths are relative to this model directory. SHA256 values were checked after the source review.

| File | SHA256 |
| --- | --- |
| `tt/factored_decode.py` | `2337245025e8d94a7b31f3585f4dbe034802af181208a2387397917fcb7c8312` |
| `tt/indexed_rows.py` | `c0d4a3e458918fa83c732f30df70e2b3079b83f014fd81afa8357040457b44a9` |
| `tt/generator.py` | `174a9cc7dd8d0d9c92c4c686e4b9d3d27d7a90961b81fadaab9975e0778df444` |
| `tt/multichip_decoder.py` | `67be6f8c0082a39d9848cdb8b7c3af44b5af543c9559a20ad2a6c2e1a1f76359` |
| `tt/model.py` | `014d95f7efc827741c4a6342f49ee2f2ee51382f3cb3fd0df23bad8692acdf94` |
| `tt/token_feedback.py` | `9443b6202493677ecd6cd7b3ce8b144c61a51da94ba5698fac978df59cee20b9` |
| `tt/prefill_checkpoint.py` | `9e352c804ea30bc07b8416e38d8068400f992c341f834619fdf47bb162770325` |

## State ordering and arithmetic

The inherited batched decoder runs each active request's attention/cache update, groups the resulting mixed activations for the existing MoE, and writes the resulting residual. Factoring delays a layer's final `mlp_mix.write` until the next layer's per-slot attention segment (or the terminal segment for the last layer). That write consumes three independent persistent values: the previous residual, block result, and injection. The next attention overwrites these buffers only after consuming them. `MeshResidual.write` is pure with respect to request cache state, so this delay preserves the required dependency order.

The persistent residual, mixed activation, block, and injection buffers are BF16, matching the inherited values at these boundaries. They do not cast a previously live FP32 value down to BF16. The existing residual write still performs its selected FP32 multiply/add and BF16 boundary casts. The owner calls the original layer attention and MoE implementations: selected GDN state, QSA/KV dtype split, expert BFP4/LoFi/internal-BFP8 policy, deterministic Qwen MoE rounding, and CCL policy remain in those implementations.

The singleton bucket retains the inherited singleton FP32 routed reduction. Every bucket above one retains the native batched BF16 routed reduction. Padding repeats the last real mixed row; it does not turn an odd batch into a set of singleton reductions or widen the batched reduction. This is material because those two accepted reduction branches have different arithmetic.

Only active slots replay attention or terminal segments. Thus dormant caches are not advanced. Padding enters the stateless MoE only, and scatter writes only the valid row count. Stale working buffers for dormant slots are safe on reactivation: layer zero rewrites their live inputs before subsequent layers consume them. Cache migration moves cache/checkpoint/seed/token ownership through the generator; these working buffers remain attached to physical slots and are overwritten on the next decode.

## Indexed rows and configured shape

The native dataflow source is byte-for-byte equal to the reduced hardware diagnostic's `tests/factored_bucket_copy.py::SOURCE`; the CPU regression checks this equality. It copies standard BF16 tile faces as bytes. The row offset `((row >> 4) * 512 + (row & 15) * 16) * 2`, together with the second half-row at another 512 bytes, addresses the two 16-column faces correctly for rows zero through 31. Read/write barriers protect scratch reuse. Cores own disjoint column tiles; valid metadata contains unique destination slots. Scatter ignores padded rows.

The internal ABI requires BF16 TILE DRAM buffers and UINT32 ROW_MAJOR `[2,32]` metadata. Host validation constructs the valid index prefix and bounds it by configured capacity. Unconfigured entries in the fixed native address table repeat the last allocated slot, but valid generated metadata never addresses those entries. Corrupted metadata is outside this trusted internal ABI; this review does not claim native bounds checks against arbitrary device-memory corruption.

Configured batch three has buckets `(1,2,4)`. Its bucket-four warmup uses three valid slots, repeats slot two in the fourth packed row, and masks that row on scatter. The largest warmup covers every allocated slot, and all attention/terminal addresses are captured. The same reasoning applies to every non-power-of-two configured batch through 32. Actual-owner CPU tests cover configured batches two, three and 32, and metadata tests cover every configured size and valid count. Existing hardware diagnostics establish odd active counts inside a configured batch of 32, not a separately constructed batch-three generator. A small configured-B3 device smoke would close that narrower allocation/ABI coverage gap; no source defect was found in it.

## Preparation, metadata and trace lifetime

All per-slot persistent buffers, power-of-two packed buffers, and metadata exist before capture. Warmup visits every selected bucket and every configured slot, then generator preparation warms common sampling, feedback, remap and checkpoint variants. Request state is reset before capture. Mesh command-queue capture uses bypass recording (`tt_metal/distributed/fd_mesh_command_queue.cpp`), so recording the subsequent factored segments is not a second eager cache update after that reset.

The number of model traces is bounded by `1 + L*B + L*K + B`, where `K = ceil(log2(B))+1`. Full 48-layer B32 therefore has 1,857 model traces: one clear, 1,536 attention, 288 MoE and 32 terminal traces. The owner aliases `generator.traces`; it does not maintain a second release list. Normal request transitions retain these traces. Explicit `close()` releases them once, clears dictionaries and invalidates the cached replay plan. A later explicit reuse can recapture the already warmed owner. No per-request eviction was introduced.

`set_slots` uploads metadata only on a changed sorted membership. An unchanged membership reuses both metadata and the same immutable tuple of trace IDs. Reordered submissions with the same physical active set do not create a new binding. Each token still loops over that fixed tuple to enqueue trace replays; it does not rebuild per-op model commands, upload slot metadata, or read sampled tokens to reconstruct inputs. Changes in scheduler membership legitimately rebuild the plan and update the existing persistent metadata buffer once.

The generator's existing on-device common sampler and exact UINT32 inactive-token feedback helper remain responsible for token feedback. Factoring contains no sampler math, host argmax, or full-logits readback. Its clear operation resets the physical sampling canvas before writing active rows, so dormant logits cannot leak into newly active rows. Explicit host sampling and read-output APIs remain separate compatibility/readout paths.

## Evidence and pending acceptance coverage

The existing reduced real-layer `factored_decode_buckets_tracked_v1.json` and clean counterpart contain 44 checks each. They cover logical counts 1, 2, 3, 5, 8, 9, 17 and 32, every physical bucket, sparse slots and duplicated padding. They do not test every logical cardinality on hardware. The measured reduced clean overhead was 0.26–1.39%, including B32 300.2237 to 300.9989 ms. This is a measured tradeoff for bounded retention, not a claim of zero overhead or full-model performance. Configured B1 does not construct this owner and retains its original whole-model path.

The queued full48 B32 HF probe exercises shrinking active widths through 32 to one, a reverse slot migration, matched HF token distributions, cache ownership and dormant-state hashes, stable trace IDs/program counts, transition upload counters, and unchanged replay-plan identity. Its forced-reference decodes reload host tokens. Those checks alone cannot prove integrated full48 resident sampled feedback with `reload_inputs=False` over multiple steps. This was reported to the parent; the parent assigned an additional short resident-feedback segment, including a membership transition, to `serving_audit`. Acceptance must inspect that segment's actual result, not this source argument.

The latest existing `serving_probe_v6` validates the checkpoint/workspace/token-feedback wrapper at reduced B2, before this production factoring integration. It must not be relabeled as integrated factored evidence. Full48 memory fit with retained factored traces, final revision correctness/performance, the once-only public max-context run, qualitative blind verdict, and stage review remain separate acceptance gates. The capacity forecast is a planning estimate until the queued allocator evidence establishes actual fit. This audit neither reduces advertised context nor promotes the stage.

CPU verification run for this audit:

```bash
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_factored_decode_cpu
```

Result: five tests passed. These use CPU stand-ins for ownership, scheduling, metadata and release behavior; they are not device numerical or latency evidence.
