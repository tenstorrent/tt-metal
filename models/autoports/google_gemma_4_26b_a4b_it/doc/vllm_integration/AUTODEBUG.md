# AutoDebug: stage09 serving runtime prerequisites

Date: 2026-09-27. Scope: read-only source and installed-package inspection for
`google/gemma-4-26B-A4B-it`. No dependency installation, source implementation
change, server launch, or hardware access was performed by this investigator.
This report is the only file written.

## Finding

**No existing, ready serving runtime was found. Stage09 requires an externally
provisioned compatible vLLM/TT-plugin Python environment before serving can be
validated under the current instruction not to install dependencies.**

The reported `ModuleNotFoundError: openai` is an environment prerequisite
failure. It occurs before argument parsing, adapter loading, or TT device
initialization. Fixing only that import, or adding the cloned source directory
to `PYTHONPATH`, would not supply the other missing server dependencies or the
TT plugin entry-point metadata. There is no source-only model fix that resolves
the identified runtime prerequisite failures.

Inspected vLLM source: `/workspace/tt-metal/vllm`, commit
`5ffebf4128f81ea5cf8413175eabde52cd8c8d75`, branch
`gemma4-vllm-integration`. The main agent switched the initial clone to this
`dev` compatibility revision while this inspection was in progress; all
requirements and plugin findings below refer to the final revision.

## Direct observations and causal chain

1. `models/common/readiness_check/run_vllm_server.py:97` imports `openai`
   unconditionally. Lines 98–99 also import `requests` and `AutoTokenizer`.
   The active Python has no `openai` module or distribution. Thus even `--help`
   fails at the module import and cannot reach argument parsing.
2. The runner launches its child server with `sys.executable` at lines 221–224.
   Changing only the shell's `vllm` executable would not change the child
   interpreter used by this runner.
3. In the tt-metal root, `find_spec("vllm")` now returns a namespace whose
   `origin` is `None`, with `/workspace/tt-metal/vllm` as its search location.
   `importlib.metadata.version("vllm")` reports no installed distribution.
   This is the outer checkout directory, not the actual package at
   `vllm/vllm/__init__.py`. A non-`None` module spec alone is insufficient proof
   of a usable installation.
4. The actual server source imports `uvloop` and `fastapi` at
   `vllm/vllm/entrypoints/openai/api_server.py:17–18`. Both are absent in the
   active Python. Exposing the source package cannot satisfy these imports.
5. The active Python has no entries in either `vllm.general_plugins` or
   `vllm.platform_plugins`. The TT package declares the necessary
   `tt_model_registry` and `tt` entries at
   `vllm/plugins/vllm-tt-plugin/pyproject.toml:28–32`.

The main agent's independent failure logs are under
`readiness_vllm/runner_startup.log` and `readiness_vllm/vllm_import.log`; this
investigation confirms the missing imports and source import paths without
launching a server.

## Existing interpreter inventory

Module availability was checked with `importlib.util.find_spec`; distribution
versions and entry points were checked with `importlib.metadata`. These checks
did not import TTNN, PyTorch, or vLLM or open a device.

| Interpreter | Observed serving prerequisites | Ready to serve? |
| --- | --- | --- |
| `/workspace/tt-metal/python_env/bin/python` (3.10.19) | TTNN 0.78.0, torch 2.11.0+cpu, transformers 5.12.1, requests 2.34.2; no openai or installed vLLM/plugin; numerous missing server dependencies | No |
| `/opt/venv/bin/python` (3.10.19) | TTNN 0.78.0 metadata; no openai, torch, transformers, requests, or installed vLLM | No |
| `/usr/bin/python3` (3.10.12) | No openai, torch, transformers, requests, or installed vLLM; tt-metal directories only appear as namespaces | No |
| `/usr/local/share/uv/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10` | No openai, torch, transformers, requests, or installed vLLM | No |

Recursive searches of `/opt`, `/home/mvasiljevic`, `/workspace`, and
`/usr/local` found the above environments, pre-commit environments, and one
cached pytest environment. The pre-commit environments contain formatter/linter
packages, with no openai, vLLM, torch, or TTNN serving distribution.

### Cached packages do not provide a ready alternative

The uv archive contains cached `openai` 3.6.0 and `vllm` 0.26.0+empty, so a
claim that no copies of these packages exist anywhere would be incorrect.
They are not installed in the surveyed interpreters.

- Cached vLLM at
  `/home/mvasiljevic/.cache/uv/archive-v0/c9892qEDM7cqVefcWBT_X`
  has wheel tag `cp312-cp312-linux_x86_64`.
- Cached uvloop, msgspec, xgrammar, and outlines_core wheels also target
  CPython 3.12, whereas the usable inspected interpreters are Python 3.10.
- Cached xgrammar 0.2.3, outlines_core 0.2.14, and llguidance 1.7.6 differ
  from this pinned fork's requirements (`==0.1.29`, `==0.2.11`, and
  `>=1.3.0,<1.4.0`, respectively).
- The cached Python 3.12 environment at
  `/home/mvasiljevic/.cache/uv/archive-v0/jnSqO4AdadfG584eeTUAa`
  contains only six pytest/support distributions. Its Python symlink points
  to the absent
  `/home/mvasiljevic/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12`.
- A cached TTNN 0.75 development wheel targets CPython 3.12 and is not the
  current TTNN 0.78.0 Python 3.10 runtime.

No attempt was made to assemble a new runtime from cache paths. These artifacts
do not establish a compatible existing serving environment.

## Exact prerequisites for this source revision

The fork's installation script at
`vllm/plugins/vllm-tt-plugin/docs/install-vllm-tt.sh:1–2` installs base vLLM
with `VLLM_TARGET_DEVICE=empty`, then installs the TT plugin. Its README
requires the active tt-metal environment (`README.md:58–96`). This is a
description of the documented provision step, not an instruction executed in
this investigation. The no-install instruction prevents doing it here.

The externally supplied environment must provide:

1. A usable Python `>=3.10,<3.14` with the matching TTNN runtime and native
   libraries, plus the model's PyTorch/Transformers dependencies. Retain the
   model's established TTNN/runtime compatibility.
2. This intended vLLM revision built for the `empty` target and its complete
   `requirements/common.txt` dependency closure. `vllm/setup.py:934–935`
   selects that file for the empty target. A CUDA-target PyPI replacement is
   not an equivalent provision step; the plugin's `pyproject.toml:12–15`
   explains that distinction.
3. The matching `vllm-tt-plugin` package, including its installed entry-point
   metadata and `tblib>=3.1.0` dependency (`pyproject.toml:18,28–32`). If
   `VLLM_PLUGINS` is set, both `tt` and `tt_model_registry` must be allowed
   (`README.md:124–128`).
4. The model adapter, its TT registry mapping, accessible model weights and
   tokenizer, and the established device/mesh launch configuration. These are
   separate stage09 integration requirements; their correctness is not proven
   by resolving Python dependencies.

The direct requirements currently absent from the main Python, based on
parsing `requirements/common.txt` and comparing installed distribution
metadata, are:

```text
blake3
fastapi[standard]>=0.115.0
openai>=1.99.1
prometheus-fastapi-instrumentator>=7.0.0
lm-format-enforcer==0.11.3
llguidance>=1.3.0,<1.4.0
outlines_core==0.2.11
diskcache==5.6.3
xgrammar==0.1.29
partial-json-parser
msgspec
gguf>=0.17.0
mistral_common[image]>=1.9.1
compressed-tensors==0.13.0
depyf==0.20.0
watchfiles
ninja
pybase64
cbor2
setproctitle
openai-harmony>=0.0.3
anthropic>=0.71.0
model-hosting-container-standards>=0.1.13,<1.0.0
mcp
grpcio
grpcio-reflection
tblib>=3.1.0
```

The platform-specific grammar requirements above apply to this x86_64 host.
The same comparison found installed-version mismatches:

| Requirement | Installed |
| --- | --- |
| `pydantic>=2.12.0` | 2.9.2 |
| `lark==1.2.2` | 1.3.1 |
| `opencv-python-headless>=4.11.0` | 4.8.1.78 |

This list compares direct distribution requirements, not all transitive
dependencies or optional-extra contents. `uvloop` is independently confirmed
missing by its module spec and is imported directly by the server. Provision
must resolve the full package closure rather than only this observed list.

## Bounded follow-up and remaining uncertainty

Once an authorized, externally provisioned runtime is available, verify its
executable and package/entry-point metadata, then rerun the readiness `--help`
and server module `--help` checks in that same interpreter. After those pass,
validate the actual serving adapter with the required hardware checks.

Moving `openai` to a lazy import could improve the runner's `--help` behavior,
but it would not make serving executable and is not the runtime repair. No
such implementation change is recommended as a substitute for provision.

The source proves why the observed import failure occurs and why the inspected
environments cannot presently execute serving. It does not prove device
behavior, model correctness, adapter compatibility, server health, sampling,
qualitative output, or performance. Those remain unverified.
