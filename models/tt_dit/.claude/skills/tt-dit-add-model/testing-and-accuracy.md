# Testing and accuracy

The gate is **PCC against a CPU reference module**. Worked examples:
`tests/models/ltx/test_vae_ltx.py`, `tests/models/wan2_2/test_transformer_wan.py`.

## The pattern

```python
from ....utils.check import assert_quality

torch_model = TorchResnetBlock3D(in_channels=in_c, out_channels=out_c)   # diffusers reference
torch_model.eval()

tt_model = TTResnetBlock3D(..., mesh_device=mesh_device)
tt_model.load_torch_state_dict(torch_model.state_dict())                 # strict=True by default

with torch.no_grad():
    torch_out = torch_model(x)
tt_out = ttnn.to_torch(tt_model(tt_x), mesh_composer=...)

assert_quality(torch_out, tt_out, pcc=0.999)
```

| Element | Why it matters |
|---|---|
| `load_torch_state_dict` is **strict by default** (`layers/module.py`) | Raises on missing *or* unexpected keys, so the mapping-totality check is free. Never pass `strict=False` to make a test pass — an unconsumed key is a real bug that surfaces at e2e |
| `assert_quality` reports PCC, CCC and RMSE/σ in one line | Consistent metrics make journal numbers comparable across components. Not `torch.allclose`, not a hand-rolled PCC |
| The reference is an **imported** module | A hand-written reference is a second implementation with its own bugs; when the gate fails you can't tell which side is wrong. Pin its commit in the journal |

## Thresholds

Counted across `tests/models/ltx/` and `tests/models/wan2_2/`:

| Bar | Where |
|---|---|
| `pcc=0.999` | Default for a layer or block. Most common by a wide margin |
| `pcc=0.99` | Blocks stacking many bf16-only kernels, or a full-model forward |
| `relative_rmse=0.007 … 0.15` | Where output is consumed as an absolute value — VAE latents, decoder pixels, audio samples |

Start at `pcc=0.999`; loosen only with a reason in the journal. **Set the bar
above the precision floor** — a bf16-only kernel contributes ~PCC 0.9999 / 3e-2
relative max error per instance and compounds. A bar below the floor is an
unfixable failing gate.

PCC alone is blind to affine error: a scaled or shifted output still scores
0.9999. That is why `relative_rmse` is paired with it wherever magnitude matters.

## The tiers

Each catches what the others cannot. Add in order.

| # | Tier | Catches | Gate |
|---|---|---|---|
| 1 | Layer / block vs reference | Key-mapping and math errors, where they're cheap to localise | `pcc=0.999`, `single_device` |
| 2 | Full component vs reference, production shapes | Tiling and stitching geometry, normalization constants, whole-tree key mapping | Same bars |
| 3 | Sharded vs **unsharded TT path** | Parallelism error, isolated from port error | DP: **PCC = 1.0** bit-exact. Spatial: normal bar **+ seams sliced separately** |
| 4 | Pipeline sanity, reference-free | Blank, frozen or corrupt output where no torch reference exists | `tests/models/wan2_2/common.py` |
| 5 | Roundtrip perceptual | Vignette, dull high end — artifacts the reference shares or PCC averages away | PSNR floor (visual); PSNR **+ log-spectrogram distance** (audio) |
| 6 | Generative quality on pipeline output | Is the generated video actually good | VBench dimensions + CLIP prompt alignment |

**Tier 3 detail.** Data parallel over work units must be bit-exact — anything
less means the split is unclean. Spatial H/W sharding is not, because halo
exchange and distributed statistics reassociate the arithmetic; halo bugs
concentrate error at boundaries where a whole-tensor PCC waves it through, so
slice the seams.

**Tier 4 detail.** `check_output_sanity` asserts shape, finiteness, uint8 range,
spatial variance (blank output) and mean inter-frame delta (frozen video), with
thresholds far below any real output so they fire only on genuine corruption.
`check_first_frame_matches_seed` is the I2V analogue — decoded frame 0 must
correlate with the seed above a floor.

**Tier 6 detail.** `utils/vbench.py::assert_vbench_quality`. Calibrated
thresholds at 1088p (`tests/models/ltx/test_pipeline_ltx_distilled.py`):

```python
{"subject_consistency": 0.92, "background_consistency": 0.93,
 "motion_smoothness": 0.955, "dynamic_degree": 1.0, "imaging_quality": 0.645}
```

Two properties worth copying: **env-gated defaulting on** (`RUN_VBENCH=0`,
`RUN_CLIP=0` skip for a perf-only iteration, so CI still gates), and **a missing
dependency reports SKIPPED, never a silent pass** — `assert_vbench_quality`
raises if `vbench` is absent; the caller guards with `pytest.importorskip`. A
quality gate that silently no-ops is worse than no gate, because it reads green.

## Artifact rubric

When output is wrong but every numeric gate passes, name the artifact.

| Artifact | Likely cause |
|---|---|
| Seams at tile or patch boundaries | Halo exchange (`neighbor_pad_async` misconfigured, wrong pad width), or normalization statistics computed per-shard instead of across the shard group |
| Temporal flicker between frames | Per-frame statistics on the wrong axis (`GroupNorm3D`, `../shared/known-issues.md`), or clip-boundary stitching |
| Banding / posterization | Precision — dtype below the reference, or fidelity dropped too far |
| Uniform blur or softness | Over-aggressive fidelity reduction, or a wrong upsample path |
| Ghosting, melting, incoherent motion | Attention or RoPE — head layout, mask, or position encoding |
| Snow / speckle | Uninitialized memory, CB overflow, or halo reading garbage |
| Blank, flat or frozen | Caught earlier by tier 4 |

Seams and flicker are the two whole-tensor PCC hides best, and both are
parallelism bugs. Look at the output, not only the number.

## Production configs only

Parametrize on shapes the model ships, with ids naming the working point —
`test_transformer_wan.py` uses `5b-720p`, `14b-480p`, `14b-720p`.

Derive them from the real schedule and put the table in the journal. Product
configs frequently collapse: for a tiled VAE the tile shape is fixed and only the
*counts* move, so four product configs can share two device shapes — testing all
four tests two shapes twice.

**Sweeping invented shapes is worse than useless.** A GroupNorm hang at
`(C=128, T=5, 32×32)` cost three hangs and two resets on a shape the real
encoder never produces.

## Keeping the suite fast

Every test is a device run paid on every future regression check.

| Do | Why |
|---|---|
| Unit tests on `single_device` | Multi-device only where the test *is* about parallelism (tier 3) |
| One shape per distinct code path, not per product config | See above |
| Reuse `line_params` / `ring_params` from `utils/test.py` | Also `*_8k`, `*_req_exact_devices`, and `mesh_device_config_to_string` for readable ids |
| `skip_if_unsupported_num_links(mesh_device, num_links)` | Skip cleanly instead of failing on a mesh that cannot do it |
| Smallest shape that still exercises the path | But it must be a shape the model really produces |
| `@pytest.mark.timeout(...)` on every device test | `../shared/device-hangs.md` |
| `l1_small_size` for convs; `trace_region_size` if tracing | `../shared/known-issues.md` |

Before adding a test, ask what failure it catches that no existing test does. If
the answer is "a shape variant of something already covered", don't add it.

## When a gate fails

Bisect against the reference: instrument it to dump per-stage intermediates,
feed both the same input, find the first stage where they diverge, compare that
stage alone. Record failed hypotheses in the journal's `Failed attempts`
(`../shared/journal-protocol.md`).
