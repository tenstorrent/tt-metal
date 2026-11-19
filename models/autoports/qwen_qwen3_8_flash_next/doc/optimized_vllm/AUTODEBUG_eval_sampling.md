# AutoDebug: required eval top-k 40 cannot use the present device sampler

Inspection date: 2026-11-17. Checkout HEAD: `ad6c615433c8ed645354a28b767f65d9b1476921`; findings describe the inspected working tree. Inspection only: no source changes, device operations, server calls, resets, or hardware reproductions. One independent source audit was used. The live greedy sweep and root `AUTODEBUG.md` were left untouched.

## Leading finding and recommended intervention

**The host route is a real, correctly advertised implementation limit, not an erroneous plugin decision.** Required `lm_eval_reasoning` requests have temperature 1, top-p 0.95, top-k 40, seed 42. Qwen advertises device top-k 32, rejects non-greedy k above 32 itself, and calls a formatter that caps k at 32. Native sampling then sorts down to 32, creates a one-tile mask, and scans only two 16-element row faces. These are independent constraints. Increasing the capability, changing only the local top-k, or changing only the native K constant cannot implement this request.

**Recommended smallest bounded extension:** preserve the complete existing family for requests with effective k <= 32; add an explicitly selected, default-off **presorted64** native sampling mode. For the wider family, run stock local top-k 64, gather the four candidate sets and integer token IDs, compute stock global top-k 64 with paired IDs, and feed `[1,1,32,64]` descending values/IDs to the new mode. It bypasses the native fixed-32 sorter and performs a two-tile mask/softmax plus a generalized candidate scan. The logical k tensor remains **40**, not 64 or 32. Keep the current persistent seed progression and token feedback. A truthful initial public limit of **40**, with physical capacity 64, is narrower than advertising 64; test all newly admitted k=33..40 before publishing that capability. Advertising 64 additionally admits 41..64 and needs that coverage.

This is a source-supported implementation proposal, **not hardware acceptance or a latency claim**. No such mode exists in the current API. The tests below are required before changing the advertised limit. No plugin change is necessary for the existing top-k capability check.

## Direct observations, separated from interpretation

| Directly inspected fact | Evidence and implication |
|---|---|
| The installed eval profile explicitly requests `(T=1, p=.95, k=40, seed=42, n=1)` | `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/eval_request_profiles.json:18-23`. This is the local profile contract; this investigation did not query an eval service. |
| Qwen's public cap is 32 and its adapter copies it | `tt/generator.py:48-58`; `tt/generator_vllm.py:25-31` under this model directory. |
| The plugin returns false for a submitted non-greedy k exceeding that cap, including under mode `all` | `/home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:4028-4085`. It also independently routes seeded requests to host when `seeded_sampling_policy == "host"` (`4073-4077`). A future cap change cannot override an explicitly selected host seed policy. |
| Direct generator use also rejects k40 before formatting | `tt/generator.py:315-325`. Constructor uses `max_top_k=32`, TP4, gather axis 1 (`139-150`). |
| The common formatter hard-caps at 32 | `models/common/sampling/generator.py:717`, `838-859`; it also changes T into 1/T and greedy into `(k=1,p=0,temp=1)`. AST-only execution of the actual formatter gave `{top_k:32, temperature:1, top_p:.95, seed:42}` for the eval request. |
| Common device normalization has another cap | `models/common/sampling/tt_sampling.py:221-246`, called at construction and reset (`381`, `876`). Its cap is `self.max_top_k`; a wider formatter alone cannot override an existing capacity-32 sampler. |
| Native sampling is fixed-32 internally | Details below. The nanobind documentation explicitly limits k to `(0,32]` at `sampling_nanobind.cpp:110`. |

Interpretation supported by those facts: this request deterministically fails current device admission or follows the host path; it does not reach a correctly functioning k40 native implementation. I did not reproduce a device crash, hang, output mismatch, or current live-server route. The out-of-row reads below describe what would happen if the guards were bypassed, not an observed failure of a currently admitted request.

## Lowered native contract and concrete k32 versus k40 contrast

Paths in this section are under `ttnn/cpp/ttnn/operations/reduction/sampling/` unless written in full.

1. `sampling.hpp`, `sampling.cpp`, and `sampling_nanobind.cpp:117-129` expose values, IDs, k/p/temp, optional scalar seed, core grid, and output; there is no sorted-input or capacity option. `device/sampling_device_operation_types.hpp` contains only seed/grid attributes and tensor arguments.
2. `device/sampling_device_operation.cpp:35-92` requires BF16 TILE interleaved values, matching INT32/UINT32 ROW_MAJOR IDs, rank 4 with N=C=1, 1..32 users, and logical width a power-of-two multiple of 32. Later checks validate k's dtype/layout/shape, **not its device contents**. Therefore these structural checks do not make k40 safe. Width 40 with a padded width 64 also fails logical-width validation; pass logical width 64 to the proposed mode.
3. The factory computes Ht=ceil(users/32), Wt=W/32 and one core per user (`device/sampling_program_factory.cpp:142-161`). Qwen always supplies 32 physical rows, so Ht=1 and 32 cores. Its present gathered input has width 128 (4 shards x 32), so Wt=4. Increasing local capacity to 64 would produce width 256, Wt=8; this passes the width check but does not increase retained K.
4. Compute hardcodes `nearest32_K=32`, `logk=5`, and consequently `Kt=1` (`device/kernels/compute/sampling.cpp:469-488`). The merge tree packs only the winning value/index tile back to the left position (`333-360`). Discarded halves are not a globally sorted continuation. Emitting two tiles after changing a constant would expose stale/unmerged losers.
5. Factory passes `ids_per_batch=tile_width=32` to the writer (`491-499`). Writer calls `generate_mask(..., ids_per_batch/32, k-1)` (`device/kernels/dataflow/writer_interleaved.cpp:107-109`). The helper, `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:251-255`, takes `(k-1) % (Sk_chunk_t*32)`. **At k32 it permits 32 columns; at k40 it wraps 39 modulo 32 to 7 and permits only eight columns.** This is not fixed by a sorter change.
6. Writer waits for one probability tile and one positional-index tile (`139-147`), scans two 16-wide row-face segments (`152-166`, `175-230`), then pops one tile each (`237-241`). For row r<16 its first segment starts at 16r and second at 256+16r. At k40 the second segment has length 24, so its last eight entries belong to another row, not the second candidate tile. Row 31 even crosses the one-tile extent.

Concrete zero-based candidate-32 (rank 33) offsets, in elements:

| Physical row | Existing continued second-face scan | Correct second-tile address |
|---:|---:|---:|
| 0 | 272 | 1024 |
| 15 | 512 | 1264 |
| 16 | 784 | 1536 |
| 31 | 1024 | 1776 |

This completes the causal chain for a fake limit increase: public admission -> formatter clamp or generator rejection; bypass those -> insufficient local candidates and/or normalization clamp; bypass those too -> sorter32, wrapped mask, and incorrect writer addresses. The intervention must address all applicable layers.

## Candidate preparation using existing device operations

The sampler-only in-tree probe identifies this model's vocabulary as 248320, local width 62080 (`tests/probe_seeded_sampling.py:3-6`, `87-98`). The production constructor derives local width as `config.vocab_size // 4`; retain that runtime derivation.

| Stage | Wider-family tensor contract, per device |
|---|---|
| Input | BF16 TILE interleaved `[1,1,32,62080]`, contiguous vocab shard, TP4 `(1,4)` |
| Local stock `topk(k=64, dim=-1, largest=True, sorted=True)` | BF16 values and positional integer indices, `[1,1,32,64]` |
| Existing all-gathers along vocab dim 3 / cluster axis 1 | `[1,1,32,256]` values and local positions, replicated over TP4 |
| Integer global IDs | Cast positions to INT32, add offsets `[0,62080,124160,186240]` in matching 64-column segments, produce UINT32 IDs; no BF16 token-ID conversion |
| Greedy tie correction, then global stock `topk(k=64)` plus pairing | Correct greedy rows before truncation/reordering; descending BF16 `[1,1,32,64]` values and matching UINT32 IDs |
| Native presorted64 input | Values TILE and corresponding IDs ROW_MAJOR, both logical `[1,1,32,64]`; k/p/temp shape `[32]`, actual k40 |
| Output | Existing UINT32 ROW_MAJOR `[1,1,1,32]` `decode_tokens` destination |

The existing source already implements local top-k, both gathers, dtype conversion, and offset addition (`models/common/sampling/tt_sampling.py:1355-1448`); `_create_indices_tensors:639-658` sizes offset segments using `max_top_k`. The wide family needs segments of 64 and total width 256. Do not reuse the 128-wide offset tensor. On Qwen `Collectives` does not expose `line_all_gather`, so `_perform_all_gather:816-840` selects stock `ttnn.all_gather`; retain its axis/order. `tt/multichip_decoder.py:99-135` describes the serial CQ0 CCL owner. The similarly named decoder `gather()` is a separate path and may apply transport precision conversion; use the existing sampling gather path for integer IDs.

Why local64 suffices: any candidate omitted from a shard's best 64 has at least 64 candidates ahead of it within that shard, so cannot be a necessary strictly ranked global top64 candidate. Local32 is insufficient: put the best 40 vocabulary logits on one shard; eight of the true best 40 are dropped before the global operation. A stdlib-only check exercised the containment property over 100 synthetic four-shard cases and this local32 counterexample. Boundary ties require an explicit tie policy, discussed below.

There are two source-supported ways to preserve the integer IDs during global selection:

**A. Positional top-k plus integer gather (conservative first implementation).** Obtain `(v64, positions64)` from `topk(gathered_values,64,sorted=True)`, then `ids64 = ttnn.gather(global_ids_uint32,3,positions64)`, then untilize IDs. Positions address 0..255 and normally use UINT16. Both gather operands must share TILE layout, and the result has the index shape. UINT32 data bypasses the BF16-only codegen gather and selects native gather: `data_movement/gather/codegen/gather_codegen_supported.cpp:74-77`; `gather.cpp:434-449`; `device/gather_device_operation.cpp:35-43`, `79-108`. The native reader uses element-size-based raw reads/writes (`device/kernels/dataflow/gather_common.hpp:11-67`; `gather_reader_single_row_single_core.cpp:169-213`). For W=256 all physical offsets are far below its uint16 local-offset limit. This code path contains no host decoding/readback. Verify exact UINT32 ID preservation on hardware; BF16-only gather tests do not prove it.

**B. Stock top-k with attached UINT32 labels (potentially smaller/faster).** `topk(gathered_values,64,indices_tensor=global_ids_uint32,stable=False,sorted=True)` returns paired labels directly, avoiding the gather. Current code explicitly widens index/output dtype from caller UINT32/INT32 indices (`reduction/topk/device/topk_device_operation.cpp:29-40`, `240-274`, `440-442`), reads label tiles verbatim (`reader_create_index_tensor.cpp:85-91`), and configures 32-bit destination accumulation in the single-core factory (`topk_single_core_program_factory.cpp:404-408`). **Do not combine custom labels with stable=True**: validation rejects that combination (`196-210`). At global W=256, k64, the single-core factory applies: the multicore low-Ht floor is 1024 (`topk_constants.hpp:21-31`). Its `compute/topk.cpp:77-94` uses sequential packs. A multicore comment about corrupted high halves under UINT32 out-of-order packs (`topk_multi_core_program_factory.cpp:128-133`) is therefore not evidence of failure for this narrow route. Existing `test_topk_indices_tensor_labels_above_uint16_max` covers k32/W16384, not k64/W256 (`tests/ttnn/unit_tests/operations/reduce/test_topk.py:638-653`). A focused k64/W256 high-label test can justify choosing B instead of A.

The wide **local** top-k keeps today's route selection. On Blackhole, unconstrained BF16 width62080 with stable=False satisfies the stock large-indices composite's non-power-of-two small-k route (`reduction/topk/topk.cpp:303-363`, `614-635`); its finish stage returns paired values/indices. Do not pass global UINT32 labels over the full local vocabulary or add preallocated outputs/sub-grids merely to use option B: those options disable that composite and change the cost/route. Use labels only for the small global 256-wide selection. The global 256-wide call is below both composite and multicore thresholds and selects stock single-core top-k. Check both selection stages with real k64 on the parent's target hardware.

## Exact presorted64 native implementation contract

Proposed API: a keyword-only `presorted=False` (name illustrative), threaded through public/primitive declarations, nanobind, `SamplingParams`, validation, and factory. With `presorted=True`, initially require logical W=64, standard 32x32 tiles, and descending paired values/IDs; retain the existing dtype/layout/user/output contract. The default path and its generated program should retain current behavior. Include the mode in operation attributes/program identity; otherwise equal input shapes in sorted and ordinary calls could reuse the wrong program. No host scan is needed to validate sortedness; it is a documented caller precondition established by `sorted=True` stock top-k. In the wide family, apply the existing greedy-row tie correction to gathered candidates **before** the global top-k. Applying that boost after the final sort can destroy descending order and make a presorted k1/p0 row choose the wrong tied token. The ordinary family keeps its existing correction-before-native-sort order.

In the new compute branch, keep random generation first, bypass `top_k`, copy two input value tiles in order into the existing two-tile `values` buffer, then run current mask/temperature/softmax helpers with **Ht=1, Kt=2**. Their loops support two columns: add, scalar temperature, MAX across both tiles, exp(logit-max), SUM across both, reciprocal, and normalization. Do not softmax each tile separately or normalize all 64 before masking to the requested k40. The writer supplies a two-tile mask (`Sk_chunk_t=2`, `cur_pos=k-1`); the shared helper already supports this without changing physical tile width. At k40 tile0 is fully unmasked and tile1 permits its first eight columns.

Prefer a new-mode writer that loops candidate ordinal `c=0..k-1`. For physical row r and candidate c, the standard tiled **element** offset is:

```
E(r,c) = 1024*(c//32) + 512*(r//16)
       + 256*((c%32)//16) + 16*(r%16) + c%16
```

Thus four 16-element segments begin at b, b+256, b+1024, b+1280, where b=512*(r//16)+16*(r%16). BF16 probability byte offset is `2*E`. With already paired/sorted IDs the result is directly `final_indices[c]`; that row begins at `r * 64 * 4` bytes in the staged ROW_MAJOR ID buffer. This eliminates the sort-position indirection in the new branch. An alternative minimal-copy implementation retains reader-generated identity index tiles, copies both to `output_ind`, then reads identity indices at E(r,c); use matching two-tile waits/pops and correct index element width. Do not feed full global UINT32 IDs through the old UInt16 sort-index buffers.

Preserve existing semantics: multiply logits by formatted 1/T, scan descending probabilities until cumulative mass **>** p (retaining the crossing candidate), renormalize that prefix, and draw using the existing three-digit 24-bit threshold. k40 remains 40 throughout. k must be in 1..64 for this native mode; a writer-side bounds assertion or equivalent device guard is useful because tensor-content bounds are not host-validated. Qwen must reject unsupported raw/unbounded non-greedy k before common clamping. Preserve its penalties/logprob restrictions.

### Per-core producer/consumer ledger for the new mode

The table instantiates physical users=32, W=64, Ht=1, Kt=2. It uses the optional identity-index variant so every existing stream is accounted for. With direct ordinal ID lookup, omit both identity streams and their bindings consistently.

| Stream | Existing factory capacity | Required new-mode traffic |
|---|---:|---|
| Input values / generated positional indices | 4 tiles each | Reader pushes 2 each; compute consumes 2 each |
| `values` | 2 tiles | Copy produces 2; mask-add, temperature and exp each pop/re-push 2; normalization consumes 2 |
| `output_ind` | 2 tiles | Copy produces 2; writer waits/pops **2** |
| `topk_mask` | 4 tiles | Writer produces 2; compute waits and finally pops 2 |
| `local_vals` | 2 tiles | Compute produces 2; writer waits/pops **2** |
| `cur_max` | 1 tile | Reduce across 2 columns produces 1; exp reuses it; final pop 1 |
| `cur_sum` | 1 tile | Reduce produces 1; reciprocal pop/re-push 1; normalization consumes 1 |
| MAX and SUM scalers | 1 tile each | Writer produces 1 each; compute finally pops 1 each |
| `rand_tile` | 1 tile | Compute produces 1; writer consumes 1 |
| `temp` | 1 entry | Writer broadcasts/pushes 1; compute consumes 1 |
| `final_indices_rm` | 32 row sticks | Reader pushes 32; writer waits/pops 32; each stick is 256 bytes, total 8192 bytes per core |

Evidence: factory `188-246`; reader `38-61`; compute helpers `66-193`, `420-455`, cleanup `505-517`; writer `139-150`, `237-241`. Retain pop-before-reserve in the in-place two-tile loops: the buffer starts full. Two rotations restore tile order; a reserve-before-pop rewrite could deadlock. The scalar helper loads both destination tiles before popping both. Remove or leave unused the sorter-transpose buffers only with coherent factory bindings. Do not leave a reader producing to a stream the new compute branch no longer drains.

`generate_mask` uses its read pointer as a copy source while fill helpers use its write pointer, but its `PNHt=1, Sk_chunk_t=2` path directly fills both tiles and needs no copy branch. This discrepancy is **not** a demonstrated blocker for this proposal. The two-tile ledger is source-feasible; descriptor binding validation, pack/reconfiguration behavior, reductions, and repeated trace completion still require hardware proof.

## Preserve the canonical family and state ownership

Do not construct every request's sampler at max_top_k64. That would double the local candidate/gather widths on ordinary <=32 requests and change the live greedy path. Establish both required physical configurations before any live trace, then select a family from authoritative request metadata. A small optional wide TTSampling component alongside the existing component is one implementation; duplicating the entire generator, cache, or request RNG owner is unnecessary. The precise Python factoring can vary, but these invariants cannot:

- Keep the current family32 local32 -> gather128 -> native sorter32 pipeline for eligible <=32 work, including Qwen's current `allow_force_argmax=False` choice. Do not enable force-argmax as part of this fix.
- Logical/public request limit and physical candidate width are separate. Make formatter capacity a backward-compatible keyword defaulting to 32; wider Qwen requests supply the supported capacity explicitly. Use physical64 for local candidates/offsets/global selection. Keep the validated raw k40 unchanged after formatting. Never interpret unbounded k as a restricted64 distribution in Qwen.
- Merge per-slot metadata before family selection. For a mixed launch containing k40 and k20, one wide program can apply each row's own k. Do not infer the family from only the newly updated row or from incoming unscattered lists. Consider **active slots** when preserving the fast path: a retired slot retaining k40 must not permanently force wide sampling.
- All 32 physical cores still execute, including inactive slots. If choosing family32 from active slots, its device k buffer must be safe for **every** physical row. An inactive stale k40 in a shared parameter buffer would violate the native32 contract even though feedback later restores its output token. Use family-specific normalized inputs or neutral inactive rows; retain authoritative request metadata for later reactivation. Do not silently clamp an active k40 row by selecting family32.
- `set_sampling_params` is also invoked on validation-only `SimpleNamespace` objects before mutations (`tt/generator.py:861-866`, `1040-1046`). Any new family/cap lookup must work there too; adding an instance field lookup without extending those validators breaks prefill/decode validation. Capability changes must also reach `generator_vllm.py`'s class-dictionary copy on fresh import.

### Persistent RNG and trace/input lifetime contract

`tt/generator.py:154-157` binds `DeviceSeedState` to the common sampler's persistent `[32]` UINT32 ROW_MAJOR DRAM seed tensor. `tt/device_seed.py:22-27`, `85-92`, `144-181`, `233-273` owns one persistent request state row per physical slot, an update buffer, active mask and output seed sink. Seeded requests use the request seed plus an absolute draw counter through the existing SplitMix64 mapping (bounded positive device seeds 1..1000000). Inactive rows output UINT32_MAX and do not advance. Unseeded mode initializes a stream once, then uses the skip sentinel. Do not introduce a second RNG owner or advance common `SeedManager` as well.

Keep this serial CQ0 sequence inside each sampling-family trace:

```
save decode_tokens
DeviceSeedState.forward()                 # active request counters advance once
local topk -> gather values/IDs -> integer global IDs
greedy-only tie correction, if applicable  # before final sort/truncation
global topk -> paired ID preparation      # input to presorted mode is descending
manual_seed(existing persistent seed sink, existing user_ids/core grid)
sampling(..., seed=None, presorted=<family>)  # random generation is first compute work
restore inactive tokens using live active mask
copy seed_skip into the same seed sink
```

The current order is in `tt/generator.py:566-587` and `models/common/sampling/tt_sampling.py:1457-1478`. `manual_seed` must remain **after** the new global top-k/gather/tie work: SFPU work can disturb PRNG/LREG state. New sampling must generate its threshold before its own SFPU math, matching `compute/sampling.cpp:467`. Do not implement seed42 as the scalar `ttnn.sampling(seed=42)` argument: it is compiled into the kernel and reinitializes on each call (`sampling_program_factory.cpp:184-186`, `406`; compute `31-61`), rather than progressing the request stream. Manual-seed's UINT32_MAX skip behavior is explicit in `reduction/manual_seed/device/kernels/compute/manual_seed_receive_all_data.cpp:32-36`.

Both families must consume the **same persistent seed sink** written by `DeviceSeedState`, or an explicitly fixed-address copy from that sink inside each trace. Merely constructing a second sampler leaves it with a different seeds tensor. Bind the sink before preparation/capture and retain that tensor's identity for the generator lifetime. The token destination, saved-token buffer, active mask, user IDs and core assignment also remain stable.

The current sampling trace key has only `(penalties_active, force_argmax)` (`tt/generator.py:569`) and is therefore insufficient when candidate family can change. Add family32 versus presorted64 to the key; include it in common trace/precompile bookkeeping if that layer is extended (`models/common/sampling/generator.py:88-100`, `164-182`, `467-559`). k/p/temp may change in-place within a fixed family without recapture, because native writer reads the persistent tensors each replay. Different candidate width/graph needs its own captured program. Family changes must not reset request counters.

Warm both local64/global64/ID conversion/new native paths, and any copies, before any trace is live. Existing `prepare()` warms sampling before reset/synchronization and model trace preparation (`tt/generator.py:454-479`). Warmup must leave live request counters untouched or restore setup state before first real sampling; capture plus the first replay must consume exactly one logical draw, as the existing sampler-only probe checks. Keep captured I/O addresses alive and refresh their contents; do not swap new tensors into a trace's inputs. Common trace input identity checks (`models/common/sampling/generator.py:390-410`) encode this contract even though Qwen captures its own `_run_sampling` closure.

Preserve `sample_logits` and `prefill_output` as separate persistent canvases (`tt/generator.py:119-133`). `_write_prefill_last:497-503` copies a temporary last-row slice to the sampler canvas and deletes it before another replay. Do not retain that temporary, an eager warmup intermediate, or a newly allocated candidate tensor across an unrelated live trace that reserves the same address range. Keep the existing lifetime/tracking regression coverage.

Enqueue is not completion: `_sample` and model/prefill replays call `execute_trace(...,blocking=False)`. Token `to_torch`/blocking `.cpu()` or explicit synchronization establishes completion (`tt/generator.py:589-600`, `1265-1274`). Async reads return an event that must complete before inspection. Parent tests must include a final read/event wait or synchronization; successful submission alone does not validate the two-tile stream balance. Host reads of the final token are compatible with device sampling; copying full vocab logits to select that token on host is the route being removed.

### Proposed implementation touch points

- `models/common/sampling/generator.py`: optional formatter capacity; optional wide-component selection/parameter reset/warmup plumbing and trace identity if this wrapper owns the wider component.
- `models/common/sampling/tt_sampling.py`: opt-in capacity64 candidate preparation and offsets; tie correction before global top-k; ID pairing; presorted native invocation. Keep the default capacity32 branch intact.
- `models/autoports/qwen_qwen3_8_flash_next/tt/generator.py`: truthful capability and raw admission limit, formatter invocation, validation-only namespaces, setup of both configurations, shared seed binding, family-aware `_sample` trace key/dispatch and warmup. `tt/generator_vllm.py` inherits the updated capability; verify this rather than patching the plugin. `tt/device_seed.py` and `tt/token_feedback.py` should not need algorithm changes.
- Under `ttnn/cpp/ttnn/operations/reduction/sampling/`: `sampling.hpp`, `sampling.cpp`, `sampling_nanobind.cpp`; `device/sampling_device_operation.hpp`, `device/sampling_device_operation_types.hpp`, `device/sampling_device_operation.cpp`; `device/sampling_program_factory.cpp`; and its three existing kernels `device/kernels/compute/sampling.cpp`, `device/kernels/dataflow/reader_values_indices_tensor.cpp`, `device/kernels/dataflow/writer_interleaved.cpp`. Thread the optional mode through signatures/attributes, branch validation/configuration and stream bindings, bypass sorting, and generalize only the opted-in writer. A separate new kernel is optional; it would also need build/source registration as appropriate.
- Existing stock top-k, integer gather and the shared mask helper supply the needed primitives; no change to their algorithms is implied by this diagnosis. Add focused tests in the existing locations below.

## Alternatives and why they rank lower

| Alternative | Assessment |
|---|---|
| Advertise cap40/64 only, or clamp k40 to32 | Incorrect. Admission/formatting and native math contradict the claimed contract. |
| Increase local top-k only | Necessary candidate coverage, insufficient final sampling. Native still retains32 and scans wrong addresses for40. |
| Direct native K64 extension | Possible but larger algorithmic change: merge pairs currently retain one tile. A genuine top64 merge must retain and merge two-tile groups through later 128+ candidate stages, rebuild/sort both halves, and fix mask, writer and buffer counts. A constant change does none of this. Reusing stock topk avoids another general top-k implementation. |
| Stock global topk64 + unchanged native sampling | Still returns native32. Pre-sorting does not bypass the sorter or expand its output. |
| Existing `sort` or topk with custom IDs | Useful for preparation; custom-ID topk option B may remove a gather. Neither performs the wider native draw by itself. |
| Pure device softmax -> cumsum -> threshold comparisons -> argmax/gather | Mathematically possible. Forward BF16/FP32 tile-axis cumsum exists (`reduction/accumulation/device/accumulation_device_operation.cpp:87-106`). It needs top-p prefix construction/renormalization and a new compatible per-request random threshold path. `rand` accepts a scalar seed and derives core seeds (`rand/device/rand_program_factory.cpp:94-97`), not the existing per-slot mutable seed-tensor contract. No standalone multinomial/categorical implementation was found under `ttnn/cpp/ttnn/operations`. This route creates more seed, trace and numerical interfaces than the bounded native extension. |

## Small, discriminating hardware tests for the parent

Run only after the ongoing greedy sweep releases hardware; this investigator has not run any of these. Start sampler-only, without model weights or a server. Adapt existing probes instead of starting full-model evaluation to debug the primitive. Each test must wait for actual completion.

1. **Pairing before RNG:** test global `[1,1,32,256]` topk64 using exact BF16 distinct scores and permuted UINT32 token labels including `65535,65536,124160,186240,248319`. Compare every returned `(value,label)` pair, not just sets. Test A's integer gather and, if chosen, B's custom-label topk separately. Repeat with new input tensor addresses (cache-hit rebinding) and as a fixed-input traced graph. This isolates ID corruption/order errors from sampling.
2. **Native two-tile boundaries:** presorted inputs `[1,1,H,64]`, H=1 and32; heterogeneous rows with k in `{1,16,17,31,32,33,40,48,64}` and p in `{0,.95,1}`. Assert exact deterministic cases (k1, p0), valid IDs, and second-tile reachability for k33/40/64. Include physical rows 0,15,16,31 and candidate ranks 16/17/32/33/40/48/64. Near-flat **strictly ordered BF16** rows give ranks33..40 substantial probability; repeated draws must reach them for k40,p1 and reach ranks33..38 for a suitable near-flat k40,p.95 fixture. Inspect all32 rows to catch wrong-row reads. k32 must never draw rank33+. Use a controlled fixed native seed only for primitive deterministic comparisons, not for progression acceptance.
3. **Prove masking before normalization:** use a large gap after rank40 and a second fixture with ranks41..64 close to rank40. For k40, changing only excluded ranks41..64 must not change seeded outcomes. For an already presorted fixture, change k40 to32 in the same input graph; support must shrink and the >32 tail must disappear. This discriminates the wrapped mask and a softmax over64 from correct top40 normalization.
4. **Nucleus/distribution:** use clear-margin CDF fixtures whose cutoff is before rank32, at a boundary, and beyond32; compare kept support and empirical mass against a BF16-aware FP32 host reference of the actual top-k -> softmax -> top-p -> renormalize contract. Reuse `test_sampling_distribution.py`'s statistical methodology, but add k40/64 and quantize p/1T as the device does. A few hundred 32-row calls can first detect an absent >32 tail; longer statistics are a follow-up only after correctness smokes pass. Do not require a stochastic TT sequence to equal PyTorch's different PRNG sequence.
5. **Full TP4 candidate contrast:** use the real local width62080 with synthetic logits and arrange all best40 on one shard, then straddle shard and candidate boundaries on all four shards. Verify global value/ID pairs before sampling. Old local32 must fail this control; local64/global64 must retain all best40. Include large/negative IDs only where the ID contract permits them; real vocabulary IDs here are nonnegative UINT32. Verify TP rank replicas agree.
6. **Persistent seed/trace lifecycle:** extend `tests/probe_seeded_sampling.py`, which already exercises actual `QwenGenerator._sample` on synthetic logits. Test seed42 reset/repeat, first capture versus cached replay, one counter increment per active logical draw, inactive rows retaining counters/tokens, pause/resume, permutation, request release/reuse, and first prefill sample followed by decode. Counter starts must match current absolute-position convention: prefill initializes to prompt end (`tt/generator.py:894-897`), decode reset to `start_pos+1` (`1077-1082`). Check counters on all four devices. Alternate family32 -> family64 -> family32 without resetting requests, then reset and repeat the **same per-step parameter schedule** against cached traces. Compare counters to continuous request progression, not output sequences from different k distributions. For a direct cross-family output control, use identical strict scores with all probability supported within the first32 candidates and identical seeds/counters. Include active32 with an inactive stale-k40 slot.
7. **Trace allocator/input lifetime:** enable the existing trace allocation tracker including program cache; warm both families, then alternate parameter updates, prefill row assembly, and sampling replay. Retain multiple family traces, refresh the same persistent input addresses, and verify output changes with contents. Ensure no capture-time/eager temporary remains live across overlapping reserved trace addresses. The existing `test_prefill_sampling_lifetime_cpu.py` is useful structural coverage but does not prove accelerator completion.
8. **Admission and unchanged fast path:** CPU tests should preserve default formatter32 behavior, admit the exact eval request with k40 intact through slot scatter/merges, reject unbounded or above-public-limit non-greedy requests before mutation, and select family32 for all eligible <=32 active requests. Once primitive/lifecycle checks pass, run a small unchanged-profile serving smoke and record actual device route plus no full-logit host sampling. Verify seed policy is device-compatible. Compare canonical <=32 graph selection and warmed latency before broad eval; do not change the eval profile, plugin, precision policy, or the active sweep to manufacture a pass.

Relevant existing test files: `tests/ttnn/unit_tests/operations/reduce/test_sampling.py`, `test_sampling_distribution.py`, `test_topk.py`; `models/common/tests/test_sampling.py`, `test_tt_sampling.py`; model `tests/test_sampling_params_cpu.py`, `tests/test_prefill_sampling_lifetime_cpu.py`, `tests/probe_seeded_sampling.py`.

## Other potential issues / qualifications

- **Numerical and tie semantics are not exact host parity.** p=.95 is stored as BF16 (0.94921875), inverse temperature is BF16, exponentials/probabilities use the current device math, and the writer uses a strict `>` cutoff. Keep requested p=.95 and T=1 in metadata; do not mislabel a BF16 implementation as bit-exact FP32 host sampling. Test robust CDF margins plus boundary behavior explicitly. The existing distribution test uses k20 (`test_sampling_distribution.py:36-37`), so it is not k40 coverage.
- **Top-k boundary ties:** fixed-cardinality top-k may choose a subset of tied kth logits, whereas threshold-style HF/vLLM filters can retain all ties. The existing sampler already has this qualification. Local/global unstable selection must keep values and IDs paired; stability is not needed merely to pair them. Test strict logits for the containment proof, test ties separately, and do not claim a new all-ties host-equivalence guarantee. For mixed wide/greedy rows, the existing lowest-global-ID tie correction must occur before the final global selection if it is meant to protect that winner from truncation; >local-cap tied maxima remain the existing limitation (`tt_sampling.py:962-984`).
- **UINT32 multicore pack comment:** a real in-tree warning about another route, not a proven failure of the proposed width256 single-core label path. It should motivate a high-label test, not a headline rejection of option B.
- **Unseeded family switching:** seeded42 reinitializes from request/counter each draw. Unseeded streams instead continue hardware RNG state, which intervening SFPU programs can influence. Do not infer unseeded cross-family sequence equality from seeded replay checks. Preserve documented unseeded behavior and test lifecycle/support; exact cross-graph unseeded sequence identity is not established by this inspection.
- No relation to a greedy chunk-sweep failure or performance result is established. That work was not inspected as failure evidence and was not modified.

## Review and validation performed here

After drafting, headline claims were rechecked against current caller guards, formatter, native factory/reader/compute/writer, shared mask helper, top-k routing, and integer gather lowering. The independent audit confirmed the two-tile offset/stream ledger and corrected an overbroad concern about multicore UINT32 packing: global W256 uses a different, sequential-pack factory. The mask read/write-pointer concern and tied-distribution qualification remain secondary, not causes of current host routing.

Only cheap local static checks ran: AST extraction/execution of the actual formatter without importing TTNN; all 32x64 tiled coordinates checked for uniqueness and inverse mapping; four concrete old-writer versus second-tile addresses evaluated; 100 synthetic local64/global64 containment cases; and a local32 counterexample losing eight of global top40. All passed their stated assertions. These checks prove arithmetic/source behavior, not a running presorted64 implementation. Root report SHA256 remained `fef4b10d1921d9d62a2b4e2902dac422e15237756db7ff817088ae2ea5fb5daf` during this investigation. Implementation and hardware validation remain the parent's work.
