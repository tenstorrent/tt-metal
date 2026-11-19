# Exact-tree norm: local scratch and wider output parallelism

Profile175 measures the two hidden norms at 3.039/5.722/13.623 us and
3.334/5.833/13.389 us for fold/inverse/output. The output phase assigns all 160
independent tiles to only 32 cores, five per core. The intermediate folded
tensor and inverse tile also make unnecessary DRAM round trips; every output
core reads the same inverse DRAM page.

This candidate changes movement and ownership only:

- Allocate the 128 KiB folded tensor and 4 KiB inverse tile directly in
  interleaved L1. Existing tensor accessors generate the appropriate L1 NoC
  addresses. There is no conversion/copy operation between passes.
- Keep fold on 32 cores and inverse on one core per user, with all six source
  strings byte-for-byte unchanged.
- Assign output's 160 tiles to 80 cores, two contiguous tiles each on the current
  device. Smaller grids use up to the available core count, with balanced counts.
  Reader, compute and writer use matching runtime tile counts and page starts.
- Keep input, gamma and final BF16 output in DRAM. Explicit DRAM scratch remains
  supported for compatibility, while the default decoder path allocates L1 scratch.

Every tile executes the identical original FP32 normalization, zero-subtract,
inverse multiply, gamma multiply and explicit BF16 RNE. Fold arithmetic and
the dense inverse reduction tree are unchanged. This does not reintroduce the
reordered reduction that failed the earlier distributed norm candidate.

Expected benefit comes from reduced per-core output work and avoiding temporary
DRAM traffic. The inverse tile remains a common source and L1 NoC contention can
limit the gain. Broader core launch and L1 allocation costs also require
measurement. No speedup is claimed before hardware validation.

Frozen artifacts:

- `candidates/hidden_rmsnorm_exact_tree_l1_80.py`
- `candidates/hidden_norm_l1_80_on173.py`
- `candidates/make_hidden_norm_l1_80_candidate.py`
- `candidates/check_hidden_norm_l1_80_cpu.py`
- `candidates/hidden_norm_l1_80_cpu.json`
- `tests/run_norm_exact_l1_comparison.py`

The new device comparator is an immutable copy of the existing norm comparator
with L1 scratch allocation and the new default helper/reference. It still uses
real model gamma, stress inputs, persistent buffers and exact BF16 bit comparisons.

CPU/source checks passed: identical fold/inverse sources, identical per-output
tile arithmetic, unchanged compute configuration builder, complete unique
ownership of all 160 pages for available grids 32–110, and the original dense
tree proof. Python compilation passed. No device was accessed and no main/shared
source was edited. L1 accessor execution, exact BF16 output, trace allocation
and latency remain unverified.

Parent-serialized probe:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_norm_exact_l1_comparison \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/hidden_norm_l1_80_probe.json
```

Require zero BF16 mismatches in all 54 default cases before whole B32 short/512
and B1 testing on `hidden_norm_l1_80_on173.py`. Whole testing exercises default
L1 allocation and both hidden norms. Profile the three phases again if latency
improves. This independent helper can be integrated with the queued beta change
by replacing only the selected exact-tree norm function.
