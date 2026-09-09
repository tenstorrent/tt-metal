# Migrated run 1000 GroupNorm tests

These goldens exercise `ttnn.migrated_run1000_groupnorm` on channel-last
`(N, 1, H*W, C)` inputs. The original PyTorch reference, tolerances, test names,
and case selection are retained. `registry.py` contains the original input
taggers, supported-axis values, and exclusions as test-only metadata; importing
the original generic operation or its Python planner is not required.

The suite and its vendored `eval/` harness come from tt_ops_code_gen revision
`0aedbc44659f273b5391e7f7e16f5280eed52903`, remote run 1000. Only TTNN operation
references and the registry import location are adapted, plus repository
formatting. The support exceptions in `ttnn.operations._op_contract` retain
their original identities.

From a built checkout with its Python environment activated:

```bash
./scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/examples/migrated_run1000_groupnorm/eval/golden_tests/groupnorm_sc_N_1_HW_C -q
```

The reference Wormhole result is 3,720 passed and 10,788 structurally invalid
cases skipped. This does not establish performance or equivalence for inputs
outside the golden suite. GN environment overrides were unset in validation;
the current native integer parser has unchecked overflow and does not accept
all Python integer-string syntax. Dedicated cache/error-boundary coverage is
also a follow-up, not a result claimed by this suite.
