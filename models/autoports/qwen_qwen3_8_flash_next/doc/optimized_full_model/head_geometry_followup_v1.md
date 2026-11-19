# Fixed-policy LM-head larger-block follow-up

2026-11-14. Keep **20 storage cores / 3 readers per DRAM bank / K block2**. All measured block5/8/10 alternatives are slower for both Q1 and Q32, including projection plus split greedy. The initial allocation error was investigated and adapted block10 controls completed successfully; it was not used to reject the entire block family.

BF16 inputs, weights and outputs, HiFi4, FP32 DEST, exact math mode and packer-off remain unchanged. These are real checkpoint weights with Seed127 normalized synthetic BF16 activations, not full-model accuracy evidence (`acceptance_eligible=false`). All runs are serial; clean timing has allocation tracking disabled.

## Artifacts, commands and provenance

Each JSON records the complete command, git commit and dirty-file list, checkpoint/weight file, input construction, program geometry, compute policy, correctness and timing. The sweep does not record per-source hashes, so its git commit alone does not identify dirty source. Retain the raw JSON/CSV/log together; the [driver](../../tests/sweep_full_model_head.py) and [DRAM implementation](../../tt/dram_projection.py) explain the measured boundary. No checkpoint tensor file was hashed.

All three runs report git commit `de432e04c21c3ea53ffe13d9c6e78a96f7acb704`, checkpoint snapshot `812d96e10a0dd82b69d749e68693698cf6c36daa`, and BF16 `lm_head.weight` [248320,2560] from `model-00131-of-00131.safetensors`. Per-rank N=62080, K=2560; Q1 and Q32 both occupy one physical tile row.

| Run | Exact candidate argument (8:3:1 and native control automatically included) | Evidence | Outcome |
|---|---|---|---|
| v1 | `20:3:2,8:3:5,16:3:5,10:3:8,20:3:5,20:3:8,20:3:10` | [JSON/provenance](head_larger_blocks_v1.json), [CSV](head_larger_blocks_v1.csv), [log](head_larger_blocks_v1.log) | Aborted at warm:c10_r3_b8; no accuracy or timing rows |
| v2 | `20:3:2,8:3:5,16:3:5,20:3:5,20:3:8` | [JSON/provenance](head_larger_blocks_v2.json), [CSV](head_larger_blocks_v2.csv), [log](head_larger_blocks_v2.log) | PASS: every candidate Q1/Q32 |
| v3 | `20:3:2,20:3:8,80:3:10,40:3:10` | [JSON/provenance](head_larger_blocks_v3.json), [CSV](head_larger_blocks_v3.csv), [log](head_larger_blocks_v3.log) | PASS: every candidate Q1/Q32 |

The JSON `command` is the exact driver invocation. Reproduction requires the [serialized environment](environment.sh), the same fixed policy, an external timeout and fresh output paths. v1 completed setup for every listed geometry and warm calls before c10_r3_b8, but setup/warmup is not measured acceptance. Its c20_r3_b5/b8/b10 entries were only prepared when execution stopped; v2/v3 supply the subsequent measurements. c20_r3_b10 itself remains unexecuted, with the source capacity estimate below; its adapted c40/c80 controls are measured.

## Complete successful timing tables

All values are milliseconds per replay, 16 timed iterations after two replay warmups, nonblocking submission with boundary synchronization and no token read in the timed window. Projection includes input sharding, linear, output restoration/slicing, Q1 canvas padding and persistent output copy. The combined window adds the common split-greedy sampler trace. Q32 describes the prefill head; current batched decode invokes Q1 heads. Compare candidates to the selected20/3/2 control in the same run.

| Run | Geometry | Q1 projection | Q1 + split greedy | Q32 projection | Q32 + split greedy |
|---|---|---:|---:|---:|---:|
| v2 | native | 0.869867 | 1.348094 | 0.847973 | 1.327715 |
| v2 | c8_r3_b1 | 0.717993 | 1.192008 | 0.693935 | 1.169833 |
| v2 | c20_r3_b2 | 0.706968 | 1.181991 | 0.683234 | 1.159947 |
| v2 | c8_r3_b5 | 0.716601 | 1.195903 | 0.692946 | 1.171022 |
| v2 | c16_r3_b5 | 0.717746 | 1.195755 | 0.692588 | 1.173382 |
| v2 | c20_r3_b5 | 0.717748 | 1.198736 | 0.699330 | 1.173309 |
| v2 | c20_r3_b8 | 0.752423 | 1.251083 | 0.720371 | 1.224855 |
| v3 | native | 0.870673 | 1.347822 | 0.847767 | 1.326964 |
| v3 | c8_r3_b1 | 0.718513 | 1.192550 | 0.694205 | 1.170276 |
| v3 | c20_r3_b2 | 0.707694 | 1.181862 | 0.682797 | 1.158853 |
| v3 | c20_r3_b8 | 0.746888 | 1.246803 | 0.730263 | 1.231213 |
| v3 | c80_r3_b10 | 0.789806 | 1.270571 | 0.766453 | 1.250924 |
| v3 | c40_r3_b10 | 0.803368 | 1.286680 | 0.786539 | 1.263084 |

Every v2/v3 candidate passes both the current8/3/1 and native TT controls, with top1/top5/top100=1 and identical common/native greedy tokens on every active row. Relative to8/3/1, minimum PCC is0.9999999982309775 in v2 and0.9999999980470723 in v3; maximum absolute errors are0.0078125 and0.015625 respectively. This is not bitwise output identity or HF accuracy. Inactive Q1 canvas rows are checked at−100, sampled tokens agree across ranks and select row maxima; measured windows compile no new programs.

## Allocation diagnosis and adapted controls

The v1 exception is a clean host compile/allocation rejection, not a kernel hang: static DFB region ends at1,228,736B while the lowest live L1 allocation starts at1,124,096B (104,640B overlap). The log records orderly driver/device close.

[`_PreparedGroup.__call__`](../../tt/dram_projection.py#L210) owns the temporary activation and output L1 tensors; the group retains only DRAM weights and host configuration. [Sweep inputs/outputs](../../tests/sweep_full_model_head.py#L311) are DRAM, samplers share one CCL provider, and candidate calls return DRAM outputs. There is no retained per-candidate activation/output L1 accumulation in this path.

The [matmul factory](../../../../../ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp#L1335) gives Kt80, bank width243tiles, three workers/bank, own81columns, 11 passes and8 columns/pass. Blackhole uses two weight streams with depth3. At block8, activation DFB=80×2048=163840B; weight streams=2×3×8×8×2048=786432B; output DFB=81×2048=165888B. The reported end implies static base112576B. c10 storage is(80/10+ceil(1940/10))×2048=413696B; adding it to the reported low address gives effective ceiling1537792B. This matches1.5MiB less32768B reserved small L1 and2304B shared semaphore allocation; removing that small pool would not close the overlap.

The [allocator validation](../../../../../tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp#L3069) compares static DFB end against the global lowest occupied compute-L1 address, so moving only worker cores does not remove this guard. The following are source-derived margins at the observed allocation ceiling, not independent memory telemetry.

| Geometry | Input/output storage B/core | Static DFB end B | Estimated remaining margin B | Actual disposition |
|---|---:|---:|---:|---|
| c10_r3_b8 | 413696 | 1228736 | −104640 | v1 compile/allocation error |
| c20_r3_b8 | 206848 | 1228736 | 102208 | v2/v3 pass; slower |
| c20_r3_b10 | 206848 | 1425344 | −94400 | Not executed; adapt storage geometry |
| c40_r3_b10 | 104448 | 1425344 | 8000 | v3 pass; slower |
| c80_r3_b10 | 53248 | 1425344 | 59200 | v3 pass; slower |

Increasing storage cores preserves the selected precision and reduces each input/output shard. The successful40/80-core block10 controls establish feasibility of that adaptation; they do not rescue performance. Keep20/3/2. Other untested decompositions are not globally disproven.

## Stage status remains open

These head trials keep no new candidate and do not establish the Optimize Stop Rule. Separately, [exact integrated resident v1](exact_integrated_resident_v1.json) records103.068541ms and9.702282t/s/u for resident token-out. Its caller-read control reproduces all128 generated tokens from [selected resident v1](selected_resident_v1.json). The fixed useful-weight/KV floor6.414528ms gives6.223556%, below the strict10% requirement; stage acceptance still fails that gate. Public caller-read103.148302ms and logits-only102.513100ms are different boundaries. The new GR-down compaction candidate and its integrated outcome remain separate work.
