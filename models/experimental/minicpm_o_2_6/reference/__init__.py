import os
import pathlib

# Make the sibling directory `reference_pytorch` available as the `reference` package.
# This allows imports like `import reference.whisper_audio` to resolve to
# models/experimental/minicpm_o_2_6/reference_pytorch/whisper_audio.py.
here = pathlib.Path(__file__).parent
candidate = (here / ".." / "reference_pytorch").resolve()
if candidate.exists():
    # Prepend to package __path__ so submodule imports resolve from reference_pytorch
    __path__.insert(0, str(candidate))
