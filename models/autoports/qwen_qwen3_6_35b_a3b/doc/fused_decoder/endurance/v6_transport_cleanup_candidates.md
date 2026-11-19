# Exact-boundary transport cleanup candidates

Source-only diagnostics from frozen ab28; no hardware run by the author. The [movement audit](v6_ab28_pattern_movement_audit.md) identifies the two sites in the valid c71 decode profile.

| Snapshot | SHA256 | Change |
|---|---|---|
| [placement](v6_ab28_transport_placement.py.snapshot) | `5d8b71b7661a926d2da1749c00752cb3414802a74e592e42c6ff5e1459bf88c8` | Remove linear `_forward` postnorm's input shard and output interleave requests. Explicit ordered norm remains unchanged; it now receives and returns DRAM. |
| [indices](v6_ab28_transport_indices.py.snapshot) | `215deb3b4b96298c7f56b661cf2bd8a61f5e885164c83a6672df9c8ba2be283d` | Use `ttnn.reshape` for selector ids, retaining ROW_MAJOR through the sparse-matmul single-stick boundary. |
| [combined](v6_ab28_transport_combined.py.snapshot) | `13f78cd31929b49bd82bcf12723bccc54e9cbad19544370daafafb93e6de77e4` | Both independent changes. |

The three `*_build.json` records prove exact byte restoration by reversing only the reviewed sites, unchanged fullkind guards, unchanged setup/kernel strings, unchanged all other method ASTs, and unchanged arithmetic boundaries. No runtime source import/dynamic compilation/helper is added. The placement removal does not change non-decode prefill statements; index removal includes public singleton prefill because it also uses the inline selector. Multi-token prefill routing is unchanged. Neither candidate removes a public capability or modifies context/PCC gates.

Builder [source](../../../tests/v6_build_transport_cleanup.py), SHA `eca7d554b175c2bac84b456dd459e4ae4606c3508b1a7912991a2349b3ced242`, archived [here](v6_transport_cleanup_builder.py.snapshot). Probe [source](../../../tests/v6_transport_cleanup_probe.py), SHA `cc3bb128267b4f14eb2c3dc46e6ef9b7f7fb0cf26ac6d77a5f2195e117887c8c`. Pinned Black23.10.1/py310, autoflake2.3.1 and isort5.13.2 checks pass; Black also leaves all three emitted snapshots unchanged.

Host source preparation passed in [source setup](v6_transport_cleanup_source_setup.json), without importing Torch or TTNN. This is not a device correctness result. The device opt-in probe extracts the actual baseline/candidate adapter statements and uses the actual norm implementation. Its diagnostic AST extraction is confined to tests; emitted decoder snapshots are standalone. The full normalization adapter covers DRAM residual→placement→ordered norm→placement→DRAM output; selector adapter covers logits→exact selector→unchanged stable softmax→BF16 score and UINT16 ROW_MAJOR single-stick outputs. All intermediate layout requests are timed. Residual addition, router projection and sparse experts are outside these component measurements and still require whole-decoder checks.

Fixtures use actual tensor geometry, real L34 checkpoint postnorm gamma plus1, seeded synthetic BF16 operands and tied router logits. Norm includes public-prefill S1/33/65/127 and four decode operands; selector includes seven random/tied/ordered256-logit fixtures. They are not captured model operands and do not establish long-stream acceptance. Readback compares exact16-bit output codes, including both scores and ids. Optional timing warms each complete adapter, reports eager and traced samples, and checks output codes after every100-replay sample. Setup uploads/readback are outside timing. Fixture tensors, command and source hashes are recorded by the device invocation.

Parent-owned device command, only after the shared device is released:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_transport_cleanup_probe.py \
  --run-device --time \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_cleanup_device.json
```

The probe refuses existing output. Current standalone source auditor pins will need explicit reviewed additions only if a candidate is selected. Existing production, frozen tests and evidence are unchanged. No candidate is acceptance-eligible before component, actual-model endurance, complete-decoder timing and appropriate source audit/review.

## Parent-run device component result

The [device probe](v6_transport_cleanup_device.json), SHA256 `5252250e9120cebf5cca8c5b46079b47544dff5f2eac8971a177c2863aba4ddd`, completed and closed the device cleanly. All8norm cases (prefill S1/33/65/127 and4decode operands) and7selector cases match exact baseline output codes; scores and indices both match. Each timed variant passed500traced replays. Complete-adapter medians:

| Adapter | Eager baseline → candidate, ms | Traced baseline → candidate, ms |
|---|---:|---:|
| norm | 0.489325263 → 0.403573178 | 0.179234659 → 0.168204349 |
| selector | 0.135566434 → 0.104759401 | 0.051880768 → 0.047694058 |

This verifies the complete adapters described above, including layout movement. It is not a complete-decoder performance result or final selection. Whole-layer34stress and matched10prefill/60decode measurements remain parent-owned gates.
