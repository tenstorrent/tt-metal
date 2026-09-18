---
name: tt-dit-kernel-research
description: "Search the tt-metal repository for an existing ttnn op or kernel before writing a new one, find fusion opportunities across adjacent ops, and modify an existing op to accept a new parameter or config knob. Use whenever a models/tt_dit optimization needs an op that may not exist, when the user asks whether a fused kernel is available, mentions writing or modifying a kernel, adding a config parameter to a ttnn op, exposing a knob through nanobind, fusing two adjacent ops, or asks whether a kernel already exists — and before proposing any new kernel work for LTX, Wan, Flux, Mochi, SD3.5, Qwen-Image, Ideogram or MiniMax-H3."
---

# TT-DiT Kernel Research

Three questions, in order:

| # | Question | Usual answer |
|---|---|---|
| 1 | Does the op already exist? | Yes — ttnn is large and its naming is not always what you'd guess |
| 2 | Does it already have the knob I need? | Often yes, just not reachable from the call site the model uses. **This gap is where the cheapest wins live** |
| 3 | Parameterize an existing op, or write a new kernel? | Parameterizing, far more often than it feels like |

This runs **before** fusion work in the optimization order
(`../tt-dit-performance/optimization-levers.md`) — hand-rolling a fusion that ttnn already
provides is the most avoidable kind of wasted iteration.

**Check `../tt-dit-benchmark-profile/existing-fast-paths.md` first.** It is the curated
answer to question (1) for the patterns diffusion models actually hit, with a
"profile shows X → try Y" table and a list of ops that are bound but have no
tt_dit caller yet. Come here for anything it doesn't cover.

## Search

```bash
ls ttnn/cpp/ttnn/operations/ ttnn/cpp/ttnn/operations/experimental/
grep -rl "<concept>" ttnn/cpp/ttnn/operations --include=*.hpp
grep -rn "def(\"<name>" ttnn/cpp/ttnn/operations --include=*_nanobind.cpp   # what's bound
./python_env/bin/python -c "import ttnn; help(ttnn.<op>)"                   # what you can call today
grep -rn "ttnn.<op>\|ttnn.experimental.<op>" models/tt_dit --include=*.py   # how others call it
```

**Check `experimental/` explicitly** — `conv3d` lives there, and so does much of
what diffusion models need; a search of the stable namespace alone misses it.
Python `help()` is authoritative for what you can *call*; header grep for what
exists in C++. The delta between them is an unbound config field — one binding
line, not a kernel.

Off-tree prior art:

```bash
gh pr list --repo tenstorrent/tt-metal --search "<op> fusion" --state all --limit 30
git log --all --oneline -- ttnn/cpp/ttnn/operations/<family>/
```

Also check whether another **model** already solved it in Python composition —
`../shared/reference-models.md`.

## Anatomy of a ttnn op

```
ttnn/cpp/ttnn/operations/<family>/<op>/
├── <op>.hpp / .cpp                       # ttnn:: entry point, defaults
├── <op>_nanobind.cpp                     # Python binding — where a knob becomes visible
└── device/
    ├── <op>_device_operation_types.hpp   # config struct / operation_attributes_t
    ├── <op>_device_operation.cpp         # validation, shape inference, program selection
    ├── <op>_program_factory.cpp          # attributes → kernel args
    └── kernels/{compute,reader_*,writer}.cpp
```

Worked example: `ttnn/cpp/ttnn/operations/experimental/conv3d/`. Its
`Conv3dConfig` in `device/conv3d_device_operation_types.hpp` carries
`weights_dtype`, `output_layout`, `T_out_block`, `W_out_block`, `H_out_block`,
`C_out_block`, `C_in_block`, `dilation`, `alignment`,
`compute_with_storage_grid_size`. That struct is exactly the surface a
performance campaign tunes, and `utils/conv3d.py::get_conv3d_config` is the
tt_dit-side table that fills it in.

## Reading an op for its knobs

"What can I pass" before "what does it do with it":

| Order | File | Tells you |
|---|---|---|
| 1 | `device/<op>_device_operation_types.hpp` | The parameter surface — the config struct |
| 2 | `<op>.hpp` | Entry point and defaults |
| 3 | `<op>_nanobind.cpp` | What's reachable from Python. Diff against (1) |
| 4 | `device/<op>_device_operation.cpp` | Which combinations are legal, which silently fall back |
| 5 | `device/<op>_program_factory.cpp` | Compile-time vs runtime arg |
| 6 | `device/kernels/*.cpp` | Last, and only if the change is truly kernel-side |

## Adding a parameter

1. **Add the field** to the config struct with a default that reproduces current
   behaviour **exactly**.
2. **Validate** in `device/<op>_device_operation.cpp` — reject illegal values
   with a `TT_FATAL` naming the value. A clear rejection beats a hang.
3. **Use it** in the program factory. Decide compile-time vs runtime.
4. **Bind it** in `<op>_nanobind.cpp`, keeping the default.
5. **Expose it in tt_dit** through the model's config table
   (`register_conv3d_configs`, `register_matmul_configs`) rather than a raw
   call-site argument — one place to tune, mechanically sweepable.
6. **Rebuild**, then gate in this order: bit-exact at the default (proves you
   changed nothing for existing callers), correct at the new value, then measure.

Step 6's first check is the one people skip. A default that is not bit-exact
changes behaviour for every caller of that op in the repository, and you will
hear about it from someone else's CI.

| | Compile-time arg | Runtime arg |
|---|---|---|
| Kernel specializes | Yes | No |
| Cheap to sweep | No — recompile per value | Yes |
| Good for | Block sizes, tile counts, layout | Shapes that vary per call |

If the parameter exists so a campaign can sweep it, weigh the recompile cost.
`utils/sweep_mm_block_sizes.py` sweeps compile-time block sizes in one device
session by keeping the mesh open — copy that structure rather than restarting
the process per value.

## When a new kernel is actually the answer

All of these true: the op exists in no family including `experimental/`; no
composition of existing ops gets within a reasonable factor; the target is a
**top-3 op** in the warm window; and you have a correctness reference to gate
against.

Fusion ordering (fused op → fold into weights → fold into an existing op's
optional inputs → new kernel) is in
`../tt-dit-performance/optimization-levers.md` § 5, with the patterns in
`../tt-dit-benchmark-profile/existing-fast-paths.md`.

## Ground rules

- **Correctness gate every kernel change** against the unmodified op at the
  production shape, **before** measuring.
- **Measure both sides on the same warm window**, via `tt-dit-benchmark-profile`.
- **Prefer upstreaming to forking.** A tuned config table in
  `models/tt_dit/utils/` is fine; a forked kernel copy will drift unnoticed.
- **Record what you searched and did not find.** "Checked, does not exist" stops
  the next agent repeating the search.
