<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# tt-metal-models

The tt-metal `models/` Python tree, packaged so it can be installed rather than cloned.

`pip install ttnn` gives you the Tenstorrent runtime and its Python bindings, but not the
model implementations that run on it — those live in the `models/` directory of the
[tt-metal](https://github.com/tenstorrent/tt-metal) repository and, until now, were
reachable only by cloning the repository and putting it on `PYTHONPATH`.

That gap matters most for serving. The Tenstorrent vLLM plugin maps every registered
architecture to a dotted path rooted at `models.`, so with no `models` tree importable,
every model fails to load — and vLLM reports it only as
`Model architectures [...] failed to be inspected`, which names neither the missing
module nor the fix.

```sh
pip install tt-metal-models==<same version as your ttnn>
```

The import root stays `models`, so code that already imports
`models.tt_transformers.tt.generator_vllm` keeps working unchanged.

## Version coupling

`tt-metal-models` and `ttnn` are built from the same commit and carry the same version
string, and this package declares `ttnn==<that version>` as a hard requirement.

The coupling is real but invisible in the source: `models/` contains no version asserts
and no `ttnn.__version__` checks — it simply calls whatever `ttnn` is installed. A strict
pin is the honest expression of that. It makes skew a resolver error at install time
rather than an `AttributeError` deep inside model construction.

## What is and is not in the package

Shipped: every `.py` under `models/` outside test directories, plus the non-Python
payload that is loaded by name at runtime — per-model parameter and decoder configs
(`model_params/**/*.json`), the architecture and trace-region tables
(`model_targets.yaml`, `model_trace_region_sizes.yaml`), the prefetcher configuration,
and the device kernel sources under `models/demos/deepseek_v3_b1/`.

Not shipped: test suites, ~12 MB of sample media (`.wav`, `.png`, `.jpg`), and
documentation. See `MANIFEST.in`, which is the source of truth for the non-Python
payload.

### Known limitations

These are properties of the `models/` tree as it stands, not of the packaging. Each one
is a live upstream cleanup.

**The llama reference submodule — needs a licensing decision before release.**
`models/demos/t3000/llama2_70b/reference/llama` is a git submodule that is empty in a
normal clone. When it is empty,
`models.demos.t3000.llama2_70b.tt.generator_vllm` and
`models.demos.llama3_70b_galaxy.tt.generator_vllm` are unimportable from the built wheel.
Populating it before the build makes both import successfully — that is verified, not
assumed.

The obstacle is not technical. That submodule is Meta's llama repository, distributed
under the **Llama 2 Community License**, not Apache-2.0. Vendoring it into a wheel that
declares `License: Apache-2.0` and publishing it to a package index is a redistribution
decision with licence consequences (attribution, carrying the agreement, the acceptable
use policy, and the 700M-MAU clause). So `build_wheel.py` never populates it implicitly:
it builds without it and warns. `--require-submodule` fails the build instead, for a
release process that has made the decision to ship it.

Until that decision is made, those two architectures are unsupported from the packaged
artifact and should be documented as such.

**Vendored third-party reference code — declared, not hidden.** Beyond the llama
submodule, the tree vendors reference implementations under their upstream terms, and the
wheel redistributes them. The wheel's `License-Expression` therefore declares
`Apache-2.0 AND MIT AND LGPL-3.0-only AND LicenseRef-Kimi-K3`, and every governing text is
enumerated as a `License-File` and embedded in the wheel's `dist-info/licenses/`:

- **MIT** — the DeepSeek-V3 reference (`models/demos/deepseek_v3/reference/deepseek`),
  Z Lab's dflash (`models/demos/deepseek_v3_d_p/reference/dflash_prefill`), Motif
  (`models/tt_dit/reference/motif`), and nanoGPT
  (`models/experimental/nanogpt/reference`).
- **LGPL-3.0-only** — the EfficientDet reference
  (`models/experimental/efficientdetd0/reference`), imported by that model's shipped
  code, so it cannot be pruned without dropping the model. Redistribution of the
  unmodified source with its license text is compliant, but a copyleft component in the
  wheel is a policy decision reviewers should make consciously.
- **LicenseRef-Kimi-K3** — `models/demos/deepseek_v3_d_p/reference/kimi_k3/attn_res`,
  MIT plus a Model-as-a-Service restriction and an attribution condition; the governing
  text is `LICENSE-Kimi-K3` in that folder.

MANIFEST.in also ships each text inside the package, next to the code it covers.

**deepseek_v3_b1 kernels.** The `.cpp` kernel sources are shipped, but the Python code
refers to them by *repository-relative string literals*
(`"models/demos/deepseek_v3_b1/micro_ops/persistent_loop/kernels/..."`), which the
tt-metal C++ runtime resolves against its kernel search root, not against `__file__`.
Shipping the files is necessary but not sufficient: those ops still require the search
root to point at the installed tree. Until those paths are made `__file__`-relative
upstream, treat `deepseek_v3_b1` as unsupported from the packaged artifact.

**`tests.`/`tools.` imports outside the test suite.** 38 modules under `models/`
(mostly vision demos, not the LLM serving paths) import from the repository's top-level
`tests.` or `tools.` packages, which are not part of this distribution. Those modules
are unimportable from the wheel.

**pytest is not a serving dependency.** The demo entry points (`models/demos/**/demo/`)
and the `conftest.py` files use pytest as a CLI parametrization harness and import it at
module scope. Serving does not go through those, so pytest is an optional extra
(`pip install tt-metal-models[demos]`) rather than a hard dependency. This only holds as
long as no *library* module imports pytest at module scope; `import_matrix.py` checks
that the serving entry points import in an environment without pytest.

**`TT_METAL_HOME` overrides the installed package.** `models/tt_transformers/tt/model_config.py`
resolves its data root from `TT_METAL_RUNTIME_ROOT`, then `TT_METAL_HOME`, and only then
from `__file__`. If you have installed this package *and* have `TT_METAL_HOME` pointing
at an old checkout, per-model parameters load from that checkout — silently, and with no
version check. Unset those variables when using the packaged tree.

## Building

The build is not run in place. `models/` sits at the repository root next to the `ttnn`
distribution's own `pyproject.toml`, and a PEP 517 build cannot reach outside its project
directory — so `build_wheel.py` stages a build root (these packaging files plus a pruned
copy of `models/`) and builds from there.

```sh
pip install build setuptools setuptools-scm
python packaging/tt-metal-models/build_wheel.py --output-dir dist
```

The version defaults to the same setuptools-scm derivation the `ttnn` wheel uses, so the
two artifacts built from one commit agree. Pass `--version` to override.

### Verifying a build

`import_matrix.py` is the package's real contract. It imports every vLLM entry point in
the tree against an installed wheel and reports which ones resolve:

```sh
python packaging/tt-metal-models/import_matrix.py --wheel dist/tt_metal_models-*.whl
```

Run it in a virtual environment containing nothing but the wheel and its declared
dependencies. A matrix run inside an environment that already has extra packages
under-reports missing dependencies — a stray `pytest` in the environment is exactly how
this class of bug reaches users.

### Native packages

`build_native_packages.py` converts a built wheel into `python3-tt-metal-models` for apt
and dnf. It works from the wheel rather than from the source tree so that the file
selection has one source of truth.

```sh
# .deb -- correct from any host; Debian's dist-packages path is version-independent
python packaging/tt-metal-models/build_native_packages.py --wheel dist/tt_metal_models-*.whl --deb

# .rpm -- run this ON the distro it targets, where %{python3_sitelib} resolves correctly
python packaging/tt-metal-models/build_native_packages.py --wheel dist/tt_metal_models-*.whl --rpm
```

CI builds only the `.deb`, because it runs on Ubuntu. RPM distros install into a
*version-qualified* `/usr/lib/python3.N/site-packages`, and an rpm built with the wrong
`N` installs cleanly and is then silently unimportable — so the script refuses to guess
rather than emit a broken package. Build the `.rpm` on the target distro, or pass
`--rpm-python-minor` if you know which Python that distro ships.

These install into the system Python's `dist-packages`/`site-packages`. Note that they
are therefore invisible to a virtual environment created without
`--system-site-packages`, which is the common case for the pip flow — venv users should
install the wheel. The native packages depend on `python3` only: the `ttnn` runtime they
need is not available as a system package (the apt `tt-nn` package is the C++ library;
the Python bindings ship only in the `ttnn` wheel), so `ttnn` must come from pip even
when `models` comes from apt or dnf.

## Layout

| File | Purpose |
|---|---|
| `pyproject.toml` | Static metadata and package discovery. Namespace discovery is required. |
| `setup.py` | Build-time version and the `ttnn` pin. |
| `MANIFEST.in` | Source of truth for the non-Python payload. |
| `build_wheel.py` | Stages the tree and builds the wheel/sdist. |
| `import_matrix.py` | CI gate: imports every vLLM entry point against a built wheel. |
| `build_native_packages.py` | Builds the `.deb` and `.rpm` from a built wheel. |
