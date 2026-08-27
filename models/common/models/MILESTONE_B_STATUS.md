# Milestone B Galaxy Model Status

Status of Milestone B of `tttv2_2d_modules_plan.md`, as of **2026-08-27**, at commit `9d3ec5799ef`
on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Sources: `tttv2_2d_modules_milestone_b_work_log.md`, and the evidence packages
`tttv2_milestone_b_evidence/{reconcile,llama,qwen,coverage,signoff}/`.

> # The exit gate is NOT PASSED.
>
> **Four of the nine exit-gate lines were never measured, one is blocked upstream as well, one is
> FAIL, and one is only partly met. The three that pass are the mechanical boundary checks.** No
> numerical result of any kind — no PCC, no accuracy figure, no demo output, no functional smoke —
> has ever been produced on silicon for either model, at any tree, by any job.
>
> **The cause is infrastructure, not code.** Eleven of the 32 Galaxy boards
> (`0 1 2 3 4 5 6 7 10 11 14`) have been off the PCIe bus since 2026-08-26. Three consecutive
> device jobs — `mb-llama`, `mb-qwen`, `mb-coverage` — each had the mesh for a night, and between
> them produced two device passes and nine silicon defects before it died. That distinction matters
> for whoever schedules the repair: this is not a milestone that failed on its merits, it is a
> milestone that was not allowed to be measured.
>
> **This page does not claim otherwise, and it will not be revised into a pass without the
> measurements.** Milestone A declared its exit gate passed on 2026-08-19, was wrong, and the
> independent re-run that disproved it found two real defects the "passing" evidence had been
> masking. That correction is the most valuable artifact Milestone A produced. The way to not repeat
> it is to write `NOT REACHED` when nothing was reached.
>
> Exactly what remains is in [Pending work](#pending-work). The short version: get the mesh back,
> fetch Qwen's weights, then measure.

## Scope

Milestone B reconstructs two WH Galaxy `(8, 4)` models on top of the Milestone A 2D modules:

- `models/common/models/llama33_70b_galaxy` — Llama-3.3-70B, 80 layers;
- `models/common/models/qwen3_32b_galaxy` — Qwen3-32B, 64 layers, 64 decoupled attention heads;
- `models/common/models/galaxy` — everything Galaxy-specific but model-neutral: geometry and
  placement recipes, collective-resource plans, prefetch construction policy, the paged-KV metadata
  view, and a direct prefill/decode runner. **No transformer graph lives there.**

Plan steps 1–3 are Llama, 4–6 are Qwen, 7 is paged KV / concat-32 / prefix cache / device sampling /
long context.

## Current position

| | |
| --- | --- |
| **Exit gate** | **NOT PASSED** — 3 of 9 lines pass; 4 `NOT REACHED`, 1 `FAIL`, 1 `PARTIAL` |
| Root cause | **Infrastructure.** 21 of 32 boards enumerate; `ttnn` cannot open a cluster at all |
| Numerical results from silicon | **None**, either model, any mode, any tree |
| Device passes ever recorded | **2** — one-layer Llama bringup (once), the partition probe (5 cases) |
| Silicon defects found | **9** (`D-B1`–`D-B9`), all in Milestone B or shared 2D/Galaxy code; 8 fixed, 1 open |
| Host defects/gaps found | **7** (`D-C1`, `D-C2`, `G-C1`–`G-C3`, `F-C1`, `F-C2`) |
| Host coverage added | 162 step-7 tests + 22 adaptor tests, each in **3 fresh processes** |
| Device tests committed that have **never run** | **33** — both `test_step7_coverage_wh_galaxy.py` files |
| Boundary gates (1D, `llm_runtime`, model-named imports) | **All three clean**, re-verified by this job |
| Qwen additional blocker | Checkpoint **absent from this host** (~65 GB); a healthy mesh does not unblock it |
| Working tree | Committed on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`; never pushed |

### Mesh state, verified by this job at 2026-08-27T03:34Z

```text
ls /sys/class/tenstorrent | wc -l     21      <- authoritative
ls /dev/tenstorrent       | wc -l     32      <- misleading; the char devices persist
missing boards                        0 1 2 3 4 5 6 7 10 11 14
every cluster open                    TTDevice::is_pcie_hung - Read 0xffffffff over PCIe ID 17
```

**Do not trust `/dev/tenstorrent`.** It reported 32 all night while only 21 boards existed. This is
recorded because two jobs' worth of confusion turned on it.

Recovery attempts spent: **4 of 4 permitted**, across `mb-llama` (2) and `mb-qwen` (2). All failed.
`tt-smi -glx_reset` fails with `[Errno 6] No such device or address: '/dev/tenstorrent/7'` — the
reset path needs the node that is gone — and `tt-smi -r` fails on the same PCIe read. `mb-coverage`
correctly spent none. **This needs an IPMI power cycle or a host reboot**, which is outside what an
unattended job may do.

## Verification Status

*Qualified* means measured on real `(8, 4)` hardware, repeated in fresh processes. *Passed once* is
recorded as such and is **not** a qualification — on this hardware a single pass has proved nothing.

| Area | Host evidence | WH `(8, 4)` device evidence | Status |
| --- | --- | --- | --- |
| **Llama adaptor / weight conversion** | 9 tests, **3 fresh processes**, 0 skips. RoPE tables are Llama-3 *scaled* frequencies; `reverse_permute` ∘ the kernel's interleaved rotation ≡ HF layout ∘ `rotate_half` at `head_dim=128`; converted attention/MLP/LM-head weights reproduce unmodified HF at PCC ≥ 0.9999; real-checkpoint layer 0 converts to contract | **None required** | **Qualified on host.** Closes the RoPE-convention half of the author's ranked risk #1 |
| **Llama block** (decode + prefill 128/2048 PCC ≥ 0.99) | Config/contract wiring in `test_model_host.py` | **NOT REACHED.** Decode never reached the LM head; no PCC number was produced. No KV-cache PCC | **Not measured** |
| **Llama full model** (80 layers, demo, teacher-forced accuracy) | `test_model_host.py` builds the full transformer config against a mock mesh | **NOT REACHED.** The 80-layer model **has never been built**. One-layer model constructs, seals, resolves both CCL contexts and tears down — **PASSED once**, 109 s, real layer-0 weights | **Not measured.** One-layer bringup is `PASSED (single run — not qualified)` |
| **Qwen adaptor / 64-head geometry** | 13 tests, **3 fresh processes**. Attention rebuilt from converted tensors alone reproduces unmodified HF `Qwen3Attention` at PCC ≥ 0.9999 on a fixture with the real decoupled shape (`attention_dim/dim = 1.6`); `wo` pairing, head-concat width and `dim`-wide post-`wo` residual all pinned; Q/K-norm relayout proved non-circularly | **None** | **Qualified on host; unqualified on silicon.** Milestone A's C4/O4 gap is half-closed |
| **Qwen block** (decode + prefill PCC ≥ 0.99) | Contract wiring only | **NOT RUN.** No mesh **and** no weights | **Not measured, blocked twice** |
| **Qwen full model** (demo, teacher-forced accuracy) | Contract wiring only | **NOT RUN.** No mesh and no weights | **Not measured, blocked twice** |
| **Paged KV** | 39 tests ×3: no two slots address the same block at active 1/8/16/31/32; no idle sink lands in an active run; every block inside the pool; late capacity resolution reaches every layer for both model classes; bind/unbind transactional, idempotent, owner-only; a part-way bind leaves nothing bound | **NOT REACHED.** **Nothing in this tree has ever compared the paged and contiguous cache layouts** | **Host-decidable half proved. `D-C1` found** |
| **Concat-32** | 34 tests ×3 at lengths 128→2048 ascending: the flat token stream gives row *r* exactly its own span; every padded position is id 0; one K and one V `paged_fill_cache` per row against a one-row table slice, in row order, under a non-identity user order; `token_indices` addresses each row's last **real** token | **NOT REACHED.** No device demo has ever produced output | **PARTIAL — mechanism proved on host. `G-C1`, `G-C2`** |
| **Prefix cache / chunked prefill** | 19 tests ×3: chunk *c* starts at block `c*chunk/block_size`, stick-aligned, zero-padded; no block shared between slots; unaligned start and over-run both fail closed; every chunk table deallocated before the next is staged; both cross-request interaction cases | **NOT REACHED.** Addressing proved; **there is no PCC** | **Host-decidable half proved. `G-C3`** |
| **Device sampling** | 26 tests ×3: greedy ≡ host argmax exactly; per-slot heterogeneous k/p/T; **the D4 reciprocal pairing re-proved at T ∈ {0.25,0.5,0.8,2.0,4.0}, deliberately never at 1.0**; the slot→column map agrees across sampler, column selector and runner | **NOT REACHED** | **Host-decidable half proved. `D-C2`, `F-C1`** |
| **Long context 4K/32K/128K** | 32 tests ×3 — **capacity accounting, not smokes.** Table below | **NOT REACHED.** The three functional smokes were never run | **Arithmetic only** |
| **Repeat / cleanup** | 12 tests ×3: two identical `generate` calls produce identical tokens *and* identical plans; `close` unbinds, deallocates both tables and every K/V tensor, idempotent; a failed `open` leaves nothing bound | **NOT REACHED.** The L1 OOM itself needs real L1 and was **not reproduced** | **Host-decidable half proved. `L1` confirmed on host** |
| **Decode sub-device partition** | — | **5 passed in 12.8 s**, no checkpoint. Worker envelope is **not contiguous** (`x=4` sender column splits it); sender ∪ worker does **not** cover the compute grid | **PASSED (single run — not qualified)**, but it is the cheapest diagnostic in the tree |

Host mocks establish config, validation, ownership and failure-path behaviour. **They do not
substitute for device numerical, cache, repeat-invocation or teardown evidence** — the point
Milestone A made and then failed to apply, and the reason no row above reads "qualified" on the
strength of its host column alone.

### Long context — what was produced instead of smokes

Per device, replicated block pool (every device owns the whole pool and writes only the users its
column serves, which is what makes one page table valid on every device — so **this does not shrink
with the mesh**):

| Context | Served | Blocks/user | Pool | KV/device, Llama (80L) | KV/device, Qwen (64L) | RoPE/device | Chunks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 6 144 | 192 | 223 | **0.14 GiB** | 0.12 GiB | 6 MiB | 2 |
| 32K | 34 816 | 1 088 | 1 119 | **0.73 GiB** | 0.58 GiB | 34 MiB | 16 |
| 128K | 133 120 | 4 160 | 4 191 | **2.72 GiB** | 2.17 GiB | 130 MiB | 64 |

At 128K, Llama: ~2.3 GiB weights + 2.72 GiB KV + 0.13 GiB RoPE ≈ **5.2 GiB against a 12 GB device.**
It should fit; the risk is fragmentation and the 64 sequential chunked-prefill graphs, not the
total. **This is arithmetic over the resolved geometry, checked by tests. It is not a smoke test and
must not be read as one.**

## Defects found

Nine on first silicon (`D-B*`), two on the host that change a contract (`D-C*`). "How it hid" is the
column that teaches something.

| | Defect | How it hid | Fix |
| --- | --- | --- | --- |
| **D-B1** | `RotarySetup2D` built its prefill cos/sin copies with an on-device `ttnn.clone`, which compiles over the full compute grid and aborts under the decode sub-device manager | Lazy. It first ran *inside* `decode_forward`, after the partition was loaded — and sealing the prefetcher leaves the decode partition loaded, so there was no later safe moment either. An eager call at construction aborted too | Write both copies from the same host source. Numerically identical — a host-to-device write compiles no program and is legal under any manager. The **only** lazy device-weight loader among the six 2D modules that ran a compute op |
| **D-B2** | `recipes.rope_core_grids` took the first `rows` cores of the *whole* compute grid, placing decode cos/sin shards on `(0,0)` and `(4,0)` — both prefetch senders — plus one core in no sub-device | Never executed on hardware. `ttnn.embedding` aborts on it immediately; nothing on host models core ownership | Delegate to `_subgrid_cores`, the qualified helper the attention KV/SDPA/reduce-scatter placements already use. **Same defect shape as Milestone A D1/C1**: a grid named independently of the partition that must contain it. Old expression put 3 cores outside; new one puts 0 |
| **D-B3** | The Llama embedding's decode output was `ttnn.L1_MEMORY_CONFIG` — *interleaved*, so it round-robins over the whole grid including the sender columns where the ~55 MB global CB lives | Never executed. DRAM did **not** help either: `ttnn.embedding` takes its program grid from a *sharded* output's shard grid and only from there | Name `decode.residual_memcfg`, confining the program to the 16 worker cores placement already occupies, and making the following relocation a no-op |
| **D-B4** | `RMSNorm2D` distributed decode **deallocated its own return value**: `ttnn.to_memory_config` returns *the same* tt_metal tensor when the config already matches, and nanobind hands it back as a **new Python wrapper**, so the `is not` identity guard freed the buffer it was about to return | Pre-existing and latent for a whole milestone. `_decode_distributed` had simply never run on hardware. And the sibling host test mocks `to_memory_config` with `side_effect=[distributed, output]` — which always returns a distinct object, i.e. **precisely the case where the identity test happens to be right** | Guard both call sites on the memory config *before* the identity test, the idiom `attention_2d._place_qk` already uses. Pinned by a test **verified to fail against the pre-fix code** |
| **D-B5** | **L3 is not closed.** `attention_qkv_program_config` and `attention_wo_program_config` are still `dense_matmul_program_config` — the exact `(7,1)` grid Milestone A's L3 names — anchored at `(0,0)` across both sender columns | The Milestone B recipes moved the **MLP** to the ring/`gather_in0` form and left attention behind, and `MILESTONE_A_STATUS.md` recorded that as "Milestone B now makes the partition-compatible choice". **It was true of the MLP and false of attention**, and no host test could see the difference. Re-verified by this job at `recipes.py:708,711` | Populate `allowed_worker_cores` (ttnn grew the field for this and warns when it is unset). Legal — but **three worker columns instead of seven**, and that cost produced D-B9 |
| **D-B6** | `collectives._all_reduce` called `ttnn.reduce_scatter` + `ttnn.all_gather`, having first *validated* that the keyed resource has a persistent buffer and then never passing it anywhere | `ttnn.reduce_scatter` cannot run on this partition **at all**: its factory takes the worker `bounding_box()` and lays workers out from that rectangle's origin — and our worker bounding box spans `x=1..6`, straight across the `x=4` sender column. The op file carries its own `// interaction with subdevice needs to be investigated` | Mode-split, mirroring the **qualified** `MLP2D._all_reduce_tg`: decode → `all_reduce_async` with the persistent buffer as `buffer_tensor`; prefill keeps the plain pair |
| **D-B7** | `_relocate` reached a full-grid program factory by **all three** obvious spellings — `to_memory_config(t, cfg, dtype)` → `ttnn::prim::copy`; `ttnn.typecast`; and `to_memory_config` across shard specs differing in grid *and* width → `reshard_program_factory_generic` | Every spelling looks correct and is correct on an unpartitioned device. The residual grid (16 cores, 128 wide) → MLP ring grid (24 cores, 96 wide) hop hits the third | The explicit `sharded_to_interleaved`/`interleaved_to_sharded` pair, which run on their own shard grids and both accept `output_dtype`. **Cost: one DRAM round trip per placement hop** — a real decode-latency debt, recorded in the Milestone C brief |
| **D-B8** | The shared axis-0 all-reduce buffer took `_spec`'s **bfloat8_b** default while both models set `MLP2D`'s `decode_ccl_dtype` to bfloat16 — deliberately, so an 80-layer residual sum is never re-quantized | **The two consumers of a deliberately shared resource disagreed with the resource, and nothing on host could see it.** `all_reduce_async` sizes its CB from the data and checks it against the buffer's bank: `Cannot set circular buffer size to 65536 ... L1 buffer bank size of 34816 B`. Those are exactly the bfloat16 and bfloat8_b `[32,1024]` shards | `build_galaxy_decode_collectives` takes a `residual_dtype` parameter (a parameter, not a literal, so a model with a different residual dtype says so instead of silently mismatching) |
| **D-B9** | **OPEN.** After D-B8 widened the shared buffer (+30 kB of L1 on each of 50 worker cores), the attention dense matmul — confined by D-B5 to three columns, which raised `per_core_N` by the same factor the grid narrowed — overflows L1 by ~20 kB | It is a *coupled* defect: neither D-B5's confinement nor D-B8's widening overflows L1 alone. It could only appear after both fixes, i.e. on the last device run of the session | **Candidate only.** `in0_block_w` `gcd(k,8)`→`gcd(k,4)`, halving the in1 CB. **Host-green (390 passed); never executed on hardware.** Treat it as a hypothesis. The structural answer is the 24-core ring form, not this |
| **D-C1** | `Attention2D._validate_decode_page_table` **accepts** a prefill-shaped page table. It discriminates on row count alone and accepts any positive multiple of `users_per_column` | The modulo is *deliberate* — an L1-sharded decode table legitimately repeats the device-local batch once per core. The prefill table's device-local view is 32 rows and `32 == 4 * 8`, so it passes. The width check passes too, because the prefill table is stick-aligned and therefore *wider*, never narrower. The dtype matches. **Shape cannot separate the two cases** | **Not fixed.** The discriminator that would work is `memory_config()`, which the validator never consults. Fixing it requires changing an existing module test that asserts the 32-row acceptance — a boundary violation to report, not to commit. Proposed fix below |
| **D-C2** | `sampling_2d._seed_digest` is `blake2b("sampling2d:{seed}:{slot}")`, so **moving a seeded request to another slot changes its stream** — the opposite of the step-7 slot-stability requirement | Not hidden — *designed*. The slot is mixed in so that 32 slots handed one seed by a serving front end do not all emit the same token, and that protection is itself proved by a test. The requirement and the design are in direct conflict | **Not fixed, deliberately.** This is a decision about the serving contract — *is a seed per-request or per-(request, slot)?* — not a bug. It needs whoever owns that contract |

### D-C1 — the proposed fix, written out

In `_validate_decode_page_table`, require:

- `shape[0] == users_per_column` when the table's memory config is **interleaved**; or
- `shape[0] == users_per_column * n_cores` when it is **L1 height-sharded**, with `n_cores` read
  from the shard spec.

Then update `test_attention_2d.py::test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[32]`
to supply an L1-sharded table, and add the interleaved-32 case as a rejection. Coherent and
testable — just not any single job's to make unilaterally, and not validatable without a mesh.

**One assumption underneath it must be checked on silicon first.** `step7_harness.py` models a
distributed tensor's `.shape` as the **shard** shape, read out of
`TensorToMesh::Impl::create_tensor` and **not measured**. If `.shape` is actually the *global*
shape, the device-local-rows branch is unreachable for a correctly-mapped table and **D-C1 is not a
loophole but a total absence of validation.** The one-line check is in the Milestone C brief.

### Gaps and premise corrections

| | Item | Detail |
| --- | --- | --- |
| **G-C1** | Concat-32 does not compose with partial batches | `GalaxyDirectRunner(active_slots=k)` gives each idle slot a sink block, but `prefill_batched` refuses any runner with `active_slots != 32`, and `_recipe_identity` resolves only `SINGLE_ROW` or `CONCAT_32`. "Active batch 16" through the concat path means *32 physical rows of which 16 carry prompts*. A limitation, not a defect — but it constrains the DP=4 / `max_num_seqs 8` serving shape Milestone C must build |
| **G-C2** | An empty row is rejected one call too late | `generate` refuses an empty prompt; `prefill_batched` called directly plans `token_indices == -1` and leaves rejection to `project_prefill_logits`. The rejection *does* happen — but after the whole concatenated graph has run |
| **G-C3** | Dead guard plus a missing check | `"chunk_page_table requires a prefix/chunked recipe"` is unreachable, because a chunk table alone already selects `PREFIX_CHUNKED`. So a chunk table with **no chunk start** silently runs the chunked recipe from token 0 instead of being refused |
| **F-C1** | **Llama has no vocabulary padding** | The plan and the step-7 brief both assume both models pad. `128256 / (8 × 32) = 501` exactly, so `padded_vocab_size == vocab_size` and `invalid_vocab_mask is None`. **A Llama pass on the padded-vocab gate would be evidence of nothing.** Only Qwen pads (128 ids, masked to bf16 min, proved unsampleable at four temperatures on host) |
| **F-C2** | `tests/models/galaxy/test_plans.py` is not host-only | `ttnn.SubDevice` implicitly constructs the `MetalContext`, so a suite with no `mesh_device` fixture, a `MagicMock` mesh and no `_wh_galaxy` in its name still cannot run without a cluster. **13 of this tree's 18 host-gate failures are this, not defects.** On a healthy mesh they should pass — and if they do not, *that* is a finding |

## Known limitations, documented and accepted

<a id="l1"></a>**L1 — `Prefetcher2D.cleanup()` cannot free the global circular buffer.** Inherited
from Milestone A and **confirmed on the host** by `mb-coverage`, using the module suite's injectable
`create_global_cb`/`deallocate`: `cleanup()` clears `self._global_cb` **without ever handing it to
`deallocate`**, so afterwards the owner truthfully reports `owned_resources == ()` while the CB it
created was never freed. Two owners in one process allocate two and free neither, ~55 MB of L1 each.

The **OOM itself needs real L1 and has never been reproduced.** And the question that actually
matters — *is the teardown-ordering contract workable at model scale?* — is **unknown**, because the
80-layer model has never been built and `test_two_models_in_one_process` has never run. The honest
reading of a clean `cleanup()` is "nothing this object still owns", not "nothing is left on the
device". → **Milestone C.**

<a id="l2"></a>**L2 — an undersized `global_cb_size` is silently accepted.** Unchanged from
Milestone A; Milestone B neither worsened nor closed it. Small and self-contained. → **any time.**

<a id="l3"></a>**L3 — attention decode on the prefetch sub-device partition. STILL OPEN, now with a
precise diagnosis.** Milestone A recorded this as terminal against a `(7,1)` grid; the Milestone A/B
reconciliation then rewrote it to say Milestone B had made the partition-compatible choice. **First
silicon disproved that** (D-B5): the MLP moved to the ring form, both attention matmuls did not.
On the current build it aborts as `Illegal kernel placement ... Kernels cannot be placed on dispatch
cores!` rather than as a sub-device mismatch. Confining it with `allowed_worker_cores` makes it legal
but leaves **three worker columns of seven**, and its circular buffers then clash with the decode
activations already resident there (D-B9). **The remaining fix is the ring form**, which the recipes
already anticipate — `attention_qkv_collective_input_memcfg` is shaped for exactly those 24 ring
cores. → **Milestone C.**

<a id="l4"></a>**L4 — `ttnn.reduce_scatter` is unusable on a non-contiguous worker sub-device.** New
in Milestone B (D-B6). Its program factory lays workers out from the sub-device *bounding box*
origin, and the Galaxy worker bounding box spans the `x=4` sender column. Worked around by
mode-splitting `_all_reduce`. **A non-contiguous worker sub-device is a normal Galaxy configuration,
not an edge case**, so this is latent for every model that partitions its grid. Three sibling ops
have the same shape of bug. → **upstream tt-metal filing; Milestone C.**

<a id="l5"></a>**L5 — every sharded→sharded relocation goes through DRAM.** New in Milestone B
(D-B7), accepted for correctness. One round trip per placement hop, a real decode-latency cost.
Either the shard specs on either side of each hop are made compatible (so
`reshard_program_factory_same_width` applies), or `reshard_program_factory_generic` must respect the
loaded sub-device. → **Milestone C, measured before it is fixed.**

<a id="l6"></a>**L6 — the ring matmul config leaves `allowed_worker_cores` unpopulated.** It
auto-populates from the full compute grid and warns eight times per decode step that this "will
become a hard error in a future release". **Deliberately not changed** — a qualified path should not
be altered on a night with no way to re-qualify it — but it is the exact hazard behind D-B2 and
D-B5. → **Milestone C.**

## Pending work

### Blocking — must close before Milestone B can be signed off

**None of these is a code task. The first two are the milestone.**

<a id="p1"></a>**P1 — Restore the mesh.** Eleven boards off the PCIe bus. All four permitted
recovery attempts spent and failed; neither `tt-smi` reset path can recover a board that is not on
the bus. **Needs an IPMI tray power cycle or a host reboot.** Everything else is downstream of this.

<a id="p2"></a>**P2 — Fetch the Qwen3-32B checkpoint** (~65 GB) into `/proj_sw/user_dev/hf_data`.
The cache entry is `config.json` only. A healthy mesh does **not** unblock the Qwen accuracy gate.
Independent of P1 and on the critical path; it can start now.

<a id="p3"></a>**P3 — Test the D-B9 hypothesis on hardware.** The `in0_block_w` change is in the
tree, host-green, and has never run on silicon. **Until it holds, the Llama decode graph does not
complete a single step**, and every measurement below is unreachable.

<a id="p4"></a>**P4 — Measure the two accuracy gates.** Llama teacher-forced (top-1 ≥ 91%,
top-5 ≥ 99%) and Qwen (top-1 ≥ 89%, top-5 ≥ 97%). Neither has ever been measured. This is the
milestone's headline deliverable and it does not exist.

<a id="p5"></a>**P5 — Run step 7's device half.** 33 committed device tests have never been
executed. Paged-vs-contiguous PCC, concat-32 device isolation, prefix-cache agreement, sampling, and
the three long-context smokes.

<a id="p6"></a>**P6 — Settle the shard-shape assumption** (one line, §D-C1). Every host conclusion
that depends on a device-local shape rests on it, D-C1 most of all.

<a id="p7"></a>**P7 — Qualify the two single-run device passes.** The one-layer bringup and the
partition probe each passed **once**. Three fresh processes each.

### Deferrable — with the milestone each belongs to

**D-C1 — decode page-table validation by placement → Milestone C.** A contract decision plus a
module-test expectation change. Needs a mesh to validate and a human to approve.

**D-C2 — is a sampling seed per-request or per-(request, slot)? → Milestone C, but it needs a
product owner, not an engineer.** Resolve it *before* vLLM serving is built on top of it.

**L1 — `Prefetcher2D` global-CB ownership redesign → Milestone C.** Routed there by name. The
executor work is what will exercise repeated owner lifecycles, and the plan's own functional gate
asks for "repeated startup, serving, and cleanup without retained TT resources".

**D-A — physical-32 real-device trace → Milestone C.** Routed by name. Milestone A deferred it
because it needs a model-owned executor running a 2D model at batch 32; **that reasoning is now
demonstrably right rather than merely stated** — there was nothing on the Galaxy to trace, and there
still is not. Its cheaper half (proving the batched-prefill delegation preserves the default,
byte-for-byte, on N150/T3K with a 1D model) needs neither a Galaxy nor Milestone B.

**Galaxy CCL / `tt_ccl.py` merge evaluation → deferred by the plan until both models pass.** They
have not, so it stays deferred. Recorded so it is not lost. Two inputs are already on record: the
D3 `semaphore_cores` invariant, now enforced in `GalaxyModePlan` with an explicit
`allow_narrow_semaphore_cores` opt-out; and L4.

**L5, L6, and the three-column attention matmul → Milestone C performance work.** Measure the
baseline before fixing them.

**L2 — `global_cb_size` validation → any time.**

**Upstream tt-metal filing → Milestone C.** Four ops choose cores from the full grid or a sub-device
bounding box: `copy_default_tilized_program_factory.cpp:44`, `reshard_program_factory_generic.cpp:80`,
`reduce_scatter_program_factory.cpp:107`, `typecast_program_factory.cpp:109`.

### Not Milestone B's, but somebody's

<a id="o2"></a>**O2 — five 1D demo-contract host tests FAIL.** `deepseek_r1_distill_qwen_14b`,
`qwen2_7b`, `qwen25_7b` (`test_eval_prefill_signature_multiset_...`), `llama33_70b`
(`test_demo_resolves_central_trace_region_size_...`) and `llama32_3b`
(`test_generator_downgrades_n150_all_trace_to_decode_only`).

**Proved not caused by Milestone B, mechanically, by this job**: all five test files are
**byte-identical** to the Milestone A tip `bc6ad03bfc2`; `models/common/llm_runtime` is byte-identical
(`git diff` is 0 lines) — and `_plan_prefill_requests`, where three of the five fail, is
`llm_runtime` code; nothing outside `models/common/{models,modules,tests}/` and `tttv2*` changed at
all; and none of the five imports anything Milestone B touched.

**They were failing at the Milestone A tip too — nobody noticed, because Milestone A's exit gate
never ran them.** Verified from Milestone A's own gate log rather than from its prose: the
`1263 passed` run at `bf403d93fed`
(`tttv2_milestone_a_final_evidence/logs/host01_integrated_gate.log`) collected exactly three trees —
`tests/llm_runtime/`, `tests/modules/`, and, under `tests/models/`, **only `tests/models/galaxy/`**.
Zero of the five failing packages appear anywhere in that log.

So this exit-gate line is one Milestone B is the **first** milestone to measure, and it was red the
moment it was first looked at. The line is **FAIL as written** — Milestone B does not get to record
it as anything else — and the owner is whoever owns those 1D packages.

<a id="o1"></a>**O1 — Milestone A's own P4 (the 1D device regression matrix) is still outstanding.**
Routed to separate hardware and deliberately not run on this Galaxy host. No
`models/common/modules/**/*_1d.py` implementation file changed in Milestone B either, so no 1D
behaviour can have changed — but the evidence is still absent, and it is still Milestone A's
outstanding exit-gate line, not Milestone B's.

## Exit-Gate Result

Measured by this job at commit `9d3ec5799ef`. Every mechanical line was re-run here rather than
quoted; the evidence is in `tttv2_milestone_b_evidence/signoff/logs/`.

| Exit-gate requirement | Result | Why |
| --- | --- | --- |
| Llama teacher-forced, batch 1, prefill 512 / decode 511, top-1 ≥ 91%, top-5 ≥ 99% | **NOT REACHED** | No mesh. **Never measured by anyone, at any tree.** No number exists to report |
| Qwen teacher-forced, batch 1, sequence 512, top-1 ≥ 89%, top-5 ≥ 97% | **NOT REACHED + BLOCKED (upstream)** | No mesh **and** the checkpoint is absent from this host |
| Batch-32 direct demos valid, no cross-slot contamination | **PARTIAL** | Block ownership and logit isolation proved on host at active 1/8/16/31/32. **No device demo has ever produced output** |
| Batch-1 4K / 32K / 128K functional smokes pass | **NOT REACHED** | Capacity accounting produced instead. Arithmetic, not a smoke |
| Prefix-cached output matches uncached execution | **NOT REACHED** | Addressing proved on host; **there is no PCC** |
| No dependency imports from an existing model-named implementation package | **PASS** | 0 matches, re-run here |
| Zero changes to 1D module implementation files | **PASS** | 0 files over all 228 changed paths since `bc6ad03bfc2` |
| Zero changes to `llm_runtime` | **PASS** | 0 files; the tree is byte-identical |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL** (green: no) / **PASS** (expectations: yes) | 5 failures, real, and **not Milestone B's** — see [O2](#o2). The *expectations* half is separately verified: of the 5 test files Milestone B modified, **all 5 are 2D module or Galaxy hardware plumbing** (`_wh_galaxy_hardware.py`, `test_attention_2d.py`, `test_lm_head_2d.py`, `test_rmsnorm_2d.py`, `test_rmsnorm_2d_wh_galaxy.py`). No 1D model contract or demo-contract expectation was touched |

**3 PASS · 1 PARTIAL · 4 NOT REACHED · 1 FAIL. The gate is not passed.**

Two things this table deliberately does **not** say. It does not say the accuracy gates would have
failed — they were not measured, and nothing here is a judgement about what they would have shown.
And it does not say the FAIL is Milestone B's fault; it says the line is FAIL as written, which is
what the gate asks.

### Regression gate, re-run by this job

```sh
HF_HOME=/proj_sw/user_dev/hf_data python -m pytest -q \
    models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime
```

```text
18 failed, 2121 passed, 2059 skipped, 3276 deselected, 10 warnings, 351 errors in 1045.84s
exit=1
```

Log: `tttv2_milestone_b_evidence/signoff/logs/01_exit_gate_regression_20260827T033413Z.log`.

**This independently reproduces `mb-coverage`'s number** — it recorded
`18 failed, 2121 passed, 2059 skipped, 3276 deselected, 351 errors in 1048.36s` at the same tree.
Two jobs, two processes, identical counts. `HF_HOME` was exported for this run, so the
real-checkpoint tests ran rather than silently skipping.

The 18 failures decompose exactly as predicted, and this job checked the decomposition rather than
accepting it:

| Count | What | Verdict |
| --- | --- | --- |
| **13** | `models/common/tests/models/galaxy/test_plans.py` | **`F-C2`, not defects.** `ttnn.SubDevice` implicitly constructs the `MetalContext`, so this suite needs a cluster despite looking host-only. Milestone B *added* this file, so Milestone A never ran it and it has no healthy-mesh baseline — on a working mesh these should pass, and if they do not, **that is a finding for Milestone C** |
| **5** | the [O2](#o2) demo-contract set — `deepseek_r1_distill_qwen_14b`, `qwen2_7b`, `qwen25_7b`, `llama33_70b`, `llama32_3b` | **Real FAIL, not Milestone B's.** Byte-identical to the Milestone A tip |

All **351 errors** are cluster-open failures in `*_wh_galaxy*` device suites and the three `moe/`
device suites — the dead mesh, not the code. The log carries 453 occurrences of
`Read 0xffffffff over PCIe ID 17`.

**So the only genuine, unexplained regression risk in this gate is zero**, and the only line the
gate fails on is one Milestone B did not cause. That is not the same as the milestone passing, and
this page does not present it as such: the four `NOT REACHED` lines are what fails the gate, and no
host run can substitute for them.

### What a defect would have to look like to slip through each passing line

The brief asks this of every passing line, because that is how Milestone A's D4 and D5 survived —
greedy-only sampling could not reach a temperature bug, and uniform memory configs could not reach a
swapped pair.

- **The three boundary greps** are exact and mechanical. A defect slips through only by living in a
  file the grep does not name — e.g. a behavioural change to 1D **tests** (permitted, and Milestone B
  did share `_hf_reference.py`), or a new topology assumption inside `models/common/models/galaxy`,
  which is Milestone B's own code and so is outside every boundary gate by construction. **The
  scorecard, not the greps, is what covers that**, and it is below.
- **The `PARTIAL` batch-32 line** proves *addressing* — that no two slots can name the same block. It
  cannot reach a defect that corrupts a slot's data while addressing it correctly, which is precisely
  what a wrong shard placement or an aliased L1 read does. That class of defect is exactly what
  D-B1–D-B9 turned out to be, and every one of them was invisible to host tests.
- **`F-C1` is the live example of coverage that cannot reach a defect**: Llama's padded-vocab gate is
  *vacuous*, so a Llama pass on it would be evidence of nothing. It was found by checking the
  premise, not by running the test.

## Modularity Scorecard

Audited by this job against `bc6ad03bfc2..9d3ec5799ef`. **228 changed paths**, which decompose as
151 raw evidence logs, 15 markdown files, 8 evidence shell/probe scripts, and **54 Python files under
`models/`** — 17 new implementations, 6 modified implementations, 26 new test files and 5 modified
test files. Everything below concerns those 54.

> The plan is explicit that this is project evidence in its own right: *"Passing model tests while
> violating these boundaries does not count as a successful TTTv2 extension."* Milestone B passed no
> model tests on hardware — and the boundaries **held**. Those are independent findings and both are
> recorded.

| Required item | Evidence | Assessment |
| --- | --- | --- |
| **New 2D/model files** | **17 new implementation files, +7841 lines**: `models/common/models/galaxy/{collectives,direct_demo,direct_runner,kv_contract,plans,prefetch,recipes}.py`; the two product packages `llama33_70b_galaxy/` and `qwen3_32b_galaxy/` (`model.py`, `hf_adaptor.py`, `weight_utils.py`, `demo.py`, `__init__.py` each). **26 new test files, +9345/−84** — more test code than implementation code | Within Milestone B boundaries. No transformer graph in `galaxy/` |
| **Existing shared files changed, and why config alone was insufficient** | **6 files, +289/−20 in total** — 3.5% of the implementation diff. See the table below | Every one justified; four are single-defect fixes |
| **1D module implementation files changed** | **Zero.** `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → empty. Re-run by this job | **Required value met** |
| **Default runtime behaviours changed** | **Zero.** `models/common/llm_runtime` is **byte-identical** to the Milestone A tip — `git diff` returns 0 lines. Nothing outside `models/common/{models,modules,tests}/` and `tttv2*` changed at all | **Required value met.** Stronger than "no behaviour change": no change |
| **1D regression suites run, and their result** | `models/common/tests/llm_runtime` **1032 passed, 1 skipped**. 1D module host suites green. **The 1D *device* matrix was NOT run** — it is Milestone A's [P4](#o1), routed to separate hardware. The five 1D demo-contract failures are [O2](#o2): real, FAIL, byte-identical to Milestone A, not caused here | **Incomplete, and honestly so.** Host green; device matrix absent by design; five failures owned elsewhere |
| **Topology assumptions discovered in common code** | **Four, all found on silicon and all in shared code.** (1) `recipes.rope_core_grids` named a grid independently of the partition that had to contain it (D-B2) — the same shape as Milestone A's D1/C1. (2) `rope_2d.load_device_weights` assumed a device clone was always legal (D-B1). (3) `collectives._all_reduce` assumed `reduce_scatter` works on any sub-device (D-B6/L4). (4) `plans.py` assumed the shared all-reduce buffer's dtype (D-B8). **Plus the root fact underneath all of them: the Galaxy worker sub-device is not contiguous and sender ∪ worker does not cover the compute grid** — now measured and pinned by `test_partition_wh_galaxy.py` | Each is now a derived value or an explicit parameter, and each is pinned by a host test |
| **Did the extension stay inside module/config/model boundaries?** | **Yes.** All 17 new files are model or Galaxy-plumbing files. The 6 shared-file changes are 4 single-defect module fixes plus 2 additive Galaxy-plumbing changes. Nothing leaked into orchestration: `llm_runtime` is byte-identical, no 1D file changed, no model-named package is imported. The extension-discipline order was followed — config first (D-B3, D-B8), frozen config value second, mechanical delegation to an existing qualified helper third (D-B2, D-B5, D-B6). **The one change that would have been larger — moving attention to the ring form — was left as a recommendation rather than half-done** | **Boundary preserved** |

### The six shared files, and why config alone was insufficient

| File | Change | Why config could not express it |
| --- | --- | --- |
| `modules/rope/rope_2d.py` (+44/−2) | Prefill tables written from host instead of `ttnn.clone` | Config has no say over whether *materialisation* runs a device program |
| `modules/rmsnorm/rmsnorm_2d.py` (+13/−3) | Guard both `to_memory_config` calls on the memory config before the identity test | A self-deallocation bug. No config value changes it |
| `modules/attention/attention_2d.py` (+77/−10) | `wo` source-shape contract relaxed to `(n_heads*head_dim, dim)`; D5 argument swap corrected; page-table validation | The square `(dim, dim)` contract **cannot express** Qwen's decoupled 64-head geometry. A contract widening, isolated at the base of the stack so a Milestone A auditor finds it |
| `modules/lm_head/lm_head_2d.py` (+8/−2) | Activation width also accepts the column-local width (`dim/4`) | A device activation off the column-sharded residual stream *carries* its column shard. A strict superset; no Milestone A test changes behaviour |
| `models/galaxy/recipes.py` (in the +8130) | rope `batch_grid` via `_subgrid_cores`; `allowed_worker_cores`; smaller `in0_block_w` | The grids were **computed, not configured**; nothing upstream could pass a different one |
| `models/galaxy/collectives.py`, `plans.py`, `resources.py` (+46), `__init__.py` (+101) | Mode-split `_all_reduce`; `residual_dtype` parameter; `worker_cores` + `allow_narrow_semaphore_cores` on `GalaxyModePlan` (Milestone A's D3 invariant, promoted from a comment into enforcement) | The op itself was wrong for a non-contiguous sub-device. The dtype fix **is** the config fix — the value was previously unreachable |

### Test-discipline audit

- **No test was deleted, `xfail`ed, skipped, or had a threshold, tolerance or parametrization
  relaxed**, in any of the four jobs. Verified against each job's report and its logs.
- Four test-side corrections were made in `reconcile`, all **against correct production behaviour**:
  a fixture built `prefill_wo` at the wrong shape; a mock returned a `SimpleNamespace` where a
  pybind11 binding needs a real `CoreCoord`; a test expected a generic error message where
  production correctly raises a specific one; and a "rank-2" rejection case built a rank-2 table and
  so never exercised the rank check it claimed to.
- **33 committed device tests have never been executed** (`test_step7_coverage_wh_galaxy.py`, both
  models). This was a deliberate, contested call — `mb-qwen` argued against committing unrun device
  tests and the argument is sound. The mitigation is loudness: both files declare "This file has
  never been executed" in their module docstring, with the date and the reason. **They are not
  evidence and this page does not count them as any.**

## Reference — evidence packages

| Package | Contents | Logs |
| --- | --- | --- |
| `tttv2_milestone_b_evidence/reconcile/` | The Milestone A/B rebase, C1–C10 disposition, host gates, three re-runnable probe scripts | 30 |
| `tttv2_milestone_b_evidence/llama/` | **First silicon.** D-B1–D-B9, the partition probe, the mesh's death | 109 |
| `tttv2_milestone_b_evidence/qwen/` | Host qualification of the 64-head geometry and Q/K norm; `BLOCKED (infra)` | 18 |
| `tttv2_milestone_b_evidence/coverage/` | Step-7 host coverage, D-C1/D-C2, the capacity table | 26 |
| `tttv2_milestone_b_evidence/signoff/` | This job's independent re-verification of the exit gate | — |

Raw pytest logs are excluded from git by the repository's `*.log` ignore rule; `mb-llama`'s 109 were
force-added past it, the rest remain on the host that produced them (`wh-glx6u-05`). Every claim on
this page names the log or the command behind it.

**Next milestone:** `tttv2_milestone_c_brief.md` — written, and explicitly **not** an authorisation
to start. The plan gates Milestone C on both models passing Milestone B, and the gate is the point.
