---
name: bringup-fork-tests
description: The test-update pass for derived bring-up ops (ttnn.bringup.*, ttnn/ttnn/bringup). Records every call a model makes to a fork (shapes, dtypes, layouts, memory placement, scalar arguments) and adds one random-input test case per distinct call to that fork's tests, checked against a torch reference. Use at the end of a model bring-up (task O.1), or whenever a model starts using a fork or a fork changes.
---

# Derived-op tests: one case per call a model makes

Each fork in `ttnn/ttnn/bringup/<fork>/` carries its own tests, and every model that uses the fork adds to them. The
fork's tests then cover everything any model asked of it, and a change made for one model fails if it breaks another.

Inputs are random: `torch.randn` for floating-point data. Never load a model's real activations, goldens, KV caches
or checkpoint weights.

## 1. Record the calls

For a framework bring-up, use the target-size ladder rung that task O.1 names:

```
BRINGUP_SPEC=<spec> BRINGUP_CAPTURE_FORKS=<bringup>/results/fork_calls.json BRINGUP_RUNG=<rung> \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py \
  -p models.demos.common.bringup.testing.fork_capture
```

For a model outside the framework, run any device test of the whole model the same way: set
`BRINGUP_CAPTURE_FORKS`, and add `--no-precompile` and `-p models.demos.common.bringup.testing.fork_capture`. Pick
the test that runs the model at its serving chunk size.

`fork_calls.json` lists each distinct call once, as `{sig, count, op, args, kwargs}`:
- each tensor as `{shape` (per device), `dtype, layout, buffer, memory_layout, [shard_shape, shard_cores], mesh,
  devices}`;
- everything else by value.
`sig` names the call. Then list the calls with no case yet:

```
python -m models.demos.common.bringup.testing.fork_cases --capture <fork_calls.json> --spec <spec>   # or --model <name>
```

## 2. The test layout of a fork

```
ttnn/ttnn/bringup/<fork>/tests/
  __init__.py      empty (tests/ is a package, so reference.py imports cleanly)
  cases.py         CASES = [ {...}, ... ]  one dict per captured call, every model's, appended
  reference.py     the op's semantics in torch, including the fork's own changes
  test_<fork>.py   one parametrized test over CASES
```

Create the files if the fork has none yet; otherwise append to `cases.py` only. Never edit or remove another model's
case, and never loosen its tolerance. A case dict holds:

| key | value |
|---|---|
| `id` | `<model>-<short description>`, unique; the pytest id |
| `model`, `task`, `sig` | who made the call, and the captured signature it reproduces (the checker matches `model` + `sig`) |
| everything the call needs | tensor shapes, dtypes, layouts and memory placement exactly as captured, the scalar and enum arguments, the compute kernel config, the mesh shape |
| `seed` | the torch seed for its inputs |
| tolerance | e.g. `exact: True` for pure data movement, or `pcc` / `atol` / `rtol` for math |

Write values out literally. Do not import them from the model: the case must survive the model changing.

## 3. Inputs: random, but valid

- Floating-point data (activations, weights): `torch.randn(shape, generator=g)` with `g = torch.Generator().manual_seed(case["seed"])`, cast to the captured dtype. Scale it if
  the op needs a range, for example `randn * 0.1` to keep a SwiGLU out of saturation. Cut weights to bfp8 on device if
  the call had bfp8 weights, and compare against the reference on the same rounded values.
- Integer, index and metadata tensors (expert ids, offsets, counts, dispatch tables, page tables) must be valid for the
  op. Build them from random choices by the op's rules. For example: top-k expert ids from a random permutation per
  token, offsets and counts from a torch bincount / cumsum of those ids, the dispatch table from the mesh layout.
  When the model's own code builds such a table from the config (e.g. `ExpertMapping.create_dispatch_table`), call the
  same helper with the case's literal config values.
- An input that another op produces in the model (e.g. `offset_cumsum`'s output feeding `dispatch`) is built by the
  test in torch, not by running the model.

## 4. Reference and check

`reference.py` implements what the op computes, in torch, float32 or float64. It includes the fork's changes (e.g. the
GeluTanh activation of `unified_routed_expert_ffn`, or the no-fabric path of a 1-device dispatch axis). Take the
semantics from the op's docstring, its program factory and kernels, and the source op's existing tests. Do not take
them from the model's output.
- Data movement (dispatch, combine, gather, layout changes): compare exactly, over every valid output element. Say in
  the test which elements are don't-care (padding, slots routed elsewhere) and skip only those.
- Math: PCC plus a max relative error. Set the limit from the measured error on a few seeds, with a margin; state the
  measured numbers in a comment.
- Check once, by hand, that the test can fail: temporarily corrupt the device output (scale it by 1.01, or zero one
  row) and see the case fail. Do not commit that change.

## 5. The test file

- One mesh per session: `ttnn.open_mesh_device(ttnn.MeshShape(*mesh))` with `ttnn.FabricConfig.FABRIC_2D`, the same
  device parameters the model uses (e.g. `l1_small_size`). Group cases by mesh shape.
- Build the inputs, run `ttnn.bringup.<op>(...)` with the case's arguments, bring the result to torch, and compare it
  with `reference.py`.
- Parametrize over `CASES`, with `ids=[c["id"] for c in CASES]`.
- Run on device only through `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/<fork>/tests`, in the
  foreground.

## 6. Done when

```
python -m models.demos.common.bringup.testing.fork_cases --capture <fork_calls.json> --spec <spec> --run-tests
```

The run must report `fork_calls_uncovered == 0` and `fork_tests_failed == 0`. This is the gate of task O.1.
`--run-tests` runs every case of every fork the model uses, other models' cases included. Commit `fork_calls.json`
with the tests: it is the record of what the model needs from each fork.
