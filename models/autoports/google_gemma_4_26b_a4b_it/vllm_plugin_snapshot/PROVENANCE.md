# Gemma 4 vLLM plugin source snapshot

This directory is the runtime package from
`tenstorrent/vllm@7f72b1c6e905f5137fe3377f2e7b42738d3f271d`, copied without
source edits from `plugins/vllm-tt-plugin`.

The snapshot exists only as a CI publication workaround. The tested vLLM
commit could not be pushed to `tenstorrent/vllm`, while the Gemma qualification
workflow was required to use only the TT-Metal and tt-inference-server branches.
The tt-inference-server build installs this package from the exact TT-Metal
checkout when `pyproject.toml` is present here. It still installs the verified
upstream `vllm==0.26.0` empty-target engine recipe; this snapshot supplies only
the Tenstorrent plugin.

Source identity:

- Repository: `tenstorrent/vllm`
- Commit: `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`
- Parent: `5ffebf4128f81ea5cf8413175eabde52cd8c8d75`
- Source path: `plugins/vllm-tt-plugin`
- Runtime files copied: `README.md`, `pyproject.toml`, `setup.py`, and all
  tracked files under `src/`
- Tests were not duplicated here; the original integration commit retains its
  full host and TT test suite, and the autoport's published evidence records
  the executed results.

The snapshot can be removed after the exact vLLM commit is available from an
authorized vLLM repository and the CI workflow can pin that repository again.
