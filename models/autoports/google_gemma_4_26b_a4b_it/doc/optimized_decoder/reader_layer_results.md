# Whole-layer DRAM reader controls

All22 real-input controls pass on runtime `b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846`. **Retain the selected projection and reader policy:** direct interleaved QKV/output, shared reader1 for both sliding projections and reader2 for both full projections. No material whole-layer gain was measured. These completed reader controls do not declare overall v6 stage acceptance.

[Machine-readable results](reader_layer_results.json) bind every result/log to [the executed journal](reader_layer_v6/commands.json), [plan](reader_layer_v6/plan.json), source hashes, exact samples and setup metadata. The failed earlier `reader_layer_v5` attempt remains preserved: baseline passed, then a Python `ttnn.Shape` slice raised before the first QKV candidate. The corrected wrapper uses integer dimension indices. No failed attempt is counted among these22 passing controls.

## Measured whole-layer outcomes

Values below are medians of five synchronized host-wall samples, each averaging30 warmed trace replays at the final position4223. First-use compilation/capture, fixture preparation and HF computation are excluded. All controls use the same real4096/128 workload. Each candidate is a separate process; the samples are not an alternating paired experiment. Isolated native times are kept separate.

| Layer | Candidate | Prefill PCC | Minimum decode PCC | Whole-layer host µs | Difference vs baseline µs | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |

| 0 | [baseline](reader_layer_v6/layer0_baseline.json) | 0.9991560943 | 0.9952574859 | 796.9335 | +0.0000 | matched baseline |

| 0 | [qkv_r1_k1](reader_layer_v6/layer0_qkv_r1_k1.json) | 0.9991560943 | 0.9952997735 | 970.3033 | +173.3698 | slower than matched baseline |

| 0 | [qkv_r2_k11](reader_layer_v6/layer0_qkv_r2_k11.json) | 0.9991560943 | 0.9952991974 | 813.9359 | +17.0024 | slower than matched baseline |

| 0 | [qkv_r3_k11](reader_layer_v6/layer0_qkv_r3_k11.json) | 0.9991560943 | 0.9952991974 | 830.6246 | +33.6911 | slower than matched baseline |

| 0 | [output_r1](reader_layer_v6/layer0_output_r1.json) | 0.9991560943 | 0.9953390129 | 852.9968 | +56.0633 | slower than matched baseline |

| 0 | [output_r2](reader_layer_v6/layer0_output_r2.json) | 0.9991560943 | 0.9953390129 | 814.6452 | +17.7117 | slower than matched baseline |

| 0 | [output_r3](reader_layer_v6/layer0_output_r3.json) | 0.9991560943 | 0.9953390129 | 816.4920 | +19.5585 | slower than matched baseline |

| 0 | [shared_gate_up_r2](reader_layer_v6/layer0_shared_gate_up_r2.json) | 0.9991560943 | 0.9952574859 | 802.4066 | +5.4731 | slower than matched baseline |

| 0 | [shared_gate_up_r3](reader_layer_v6/layer0_shared_gate_up_r3.json) | 0.9991560943 | 0.9952574859 | 809.9629 | +13.0294 | slower than matched baseline |

| 0 | [shared_down_r2](reader_layer_v6/layer0_shared_down_r2.json) | 0.9991560943 | 0.9952574859 | 797.3182 | +0.3847 | no demonstrated gain; sample ranges overlap |

| 0 | [shared_down_r3](reader_layer_v6/layer0_shared_down_r3.json) | 0.9991560943 | 0.9952574859 | 800.7741 | +3.8406 | slower than matched baseline |

| 5 | [baseline](reader_layer_v6/layer5_baseline.json) | 0.9991234309 | 0.9950717323 | 832.4668 | +0.0000 | matched baseline |

| 5 | [qkv_r1_k1](reader_layer_v6/layer5_qkv_r1_k1.json) | 0.9991234309 | 0.9952449901 | 973.0462 | +140.5794 | slower than matched baseline |

| 5 | [qkv_r2_k11](reader_layer_v6/layer5_qkv_r2_k11.json) | 0.9991234309 | 0.9952781971 | 839.3746 | +6.9078 | slower than matched baseline |

| 5 | [qkv_r3_k11](reader_layer_v6/layer5_qkv_r3_k11.json) | 0.9991234309 | 0.9952781971 | 857.1210 | +24.6542 | slower than matched baseline |

| 5 | [output_r1](reader_layer_v6/layer5_output_r1.json) | 0.9991234309 | 0.9950420399 | 842.0106 | +9.5438 | slower than matched baseline |

| 5 | [output_r2](reader_layer_v6/layer5_output_r2.json) | 0.9991234309 | 0.9950420399 | 855.5988 | +23.1320 | slower than matched baseline |

| 5 | [output_r3](reader_layer_v6/layer5_output_r3.json) | 0.9991234309 | 0.9950420399 | 870.4437 | +37.9769 | slower than matched baseline |

| 5 | [shared_gate_up_r1](reader_layer_v6/layer5_shared_gate_up_r1.json) | 0.9991234309 | 0.9950717323 | 837.6792 | +5.2124 | slower than matched baseline |

| 5 | [shared_gate_up_r3](reader_layer_v6/layer5_shared_gate_up_r3.json) | 0.9991234309 | 0.9950717323 | 837.0823 | +4.6155 | slower than matched baseline |

| 5 | [shared_down_r1](reader_layer_v6/layer5_shared_down_r1.json) | 0.9991234309 | 0.9950717323 | 832.4193 | -0.0474 | no demonstrated gain; sample ranges overlap |

| 5 | [shared_down_r3](reader_layer_v6/layer5_shared_down_r3.json) | 0.9991234309 | 0.9950717323 | 832.5128 | +0.0460 | no demonstrated gain; sample ranges overlap |


Full shared down reader1 is nominally0.0474µs faster than reader2, but its samples span831.3316–833.6520µs versus baseline832.2223–832.9460µs. Full gate2/down3 is0.0460µs slower, with similarly overlapping samples. Sliding down reader2 is0.3847µs slower and also overlaps baseline. These observations do not resolve a performance improvement; they do not justify replacing the current policy or claiming statistical equivalence. No additional repeat is required to retain the already validated defaults.

## Isolated native versus integrated work

[The isolated matrix](dram_reader_results.md) holds padded shapes, activation storage, quantized operands and compute flags fixed within each K family, uses all six reader-order permutations, and validates20 native matmuls per window. It records device distributions, header-inclusive logical/physical weight GB/s, peak percentage, row/bank geometry, NoC request estimates and BRISC/NCRISC/TRISC intervals. All28 legal micro cases pass; reader1/K11 exceeds static L1 at1,641,728B versus1,572,864B. Therefore the integrated reader1 QKV control uses K1. Reader1/K1 versus reader2/K11 is a candidate-family comparison, not a reader-only speedup claim.

| Role | Isolated fastest legal native configuration | Whole-layer outcome versus selected baseline |
| --- | --- | --- |
| Sliding QKV | K11/reader2,77.7470µs | +17.0024µs |
| Full QKV | K11/reader2,75.4375µs | +6.9078µs |
| Sliding output | K16/reader3,47.4430µs | +19.5585µs; integrated reader2 is less slow at+17.7117µs |
| Full output | K16/reader1,58.1370µs | +9.5438µs |
| Sliding shared gate/up and down | Current reader1,32.9655/18.2145µs | Additional readers do not improve the layer |
| Full shared gate/up | Current reader2,21.9305µs | Readers1/3 lose |
| Full shared down | Reader3,13.6150µs versus reader2,13.9585µs | Gate2/down3 gains no resolved layer time |

The isolated trace contains one linear operation and excludes producer/consumer movement. The integrated [wrapper](../../tests/probe_optimized_reader_layer.py) includes the actual candidate boundaries:

- DRAM QKV reshards the production FP32 input, produces a width-sharded FP32 result, converts it to the existing L1 consumer layout, and slices padded channels. Common N9216 adds12.5% weight width versus sliding's selected logical8192; full is already9216. All readers share this explicit padding, whose channels remain inert.
- The existing [output adapter](../../tests/probe_optimized_output.py) retains head concat, reshards its BF16 result for DRAM matmul, then restores the L1 consumer layout and slices N3072 to logical2816. The selected interleaved projection operates at logical2816. Thus integrated comparisons include padding, input/output layout boundaries and the different matmul program. The test does not promote BF16 attention to FP32 input.
- Shared projections already use the padded DRAM contract: gate/up2816×4608 and down2112×3072. Their controls replace only one role's reader program; dtype, compute, buffers and surrounding paths remain selected defaults. Full down-reader3 explicitly retains gate-reader2.

These source-visible boundaries explain why the isolated comparison is insufficient to select a layer policy; the results measure the complete effect. They do not assign the measured differences exclusively to movement or claim a hardware-counter attribution. Additional readers also change reader placement/multicast/compute distribution; RISC duration intervals include waits.

## Current precision and correctness proof

The isolated runs used runtime169c0d97; the integrated runs use b585a21f after tail-buffer ownership and tight-cache guards changed. Both attention kinds' current baseline `precision_policy` dictionaries match all42 fields from their isolated production captures exactly, and original fixture hashes match. Every integrated candidate also preserves both selected minimal-prefill policy dictionaries exactly. Runtime, wrapper and report hashes are checked by the result summarization; no numerical-policy equivalence is inferred merely from filenames.

| Projection | Sliding | Full |
| --- | --- | --- |
| QKV | FP32×BFP8→FP32, HiFi2, FP32 destination | FP32×BFP8→FP32, LoFi, FP32 destination |
| Output | BF16×BFP8→FP32, HiFi4, FP32 destination | BF16×BFP8→FP32, LoFi, FP32 destination |
| Shared gate/up and down | BF16×BFP8→BF16, LoFi, BF16 destination/packer accumulation | BF16×BFP4→BF16, LoFi, BF16 destination/packer accumulation |

Native dtype/fidelity/reader proof comes from the isolated raw CSVs and parser checks. Integrated result metadata records actual selected weight tensors, exact compute/program settings and padding; configured output input remains BF16 at the native-attention boundary. These unprofiled whole-layer controls do not manufacture per-op device durations. Both minimal QKV policies (K8 sliding/K16 full) and full minimal output K8 remain active; the current experts, router, native attention and cache policy stay selected.

Every result passes prefill PCC0.995, all128 distinct positions4096–4223, exact repeated trace outputs, device-only prefill/decode audits and the program-cache guard. The report has129 checks because the first position is checked again; that is not129 unique tokens. Input refresh and reference checks are outside measured replay batches. [The isolated formatting-provenance incident](dram_readers_v5/provenance_incident.md) preserves the old/new probe split without rewriting measured data; it does not affect runtime or precision policy.

The installed OPT-015 reader comparison and whole-layer integration requirement is complete for these four material projection roles and both attention kinds. No reader candidate was adopted. Remaining v6 validation, final profiles/accounting, independent review and commit are tracked separately.
