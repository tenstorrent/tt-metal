"""Resolve a Hugging Face home that actually holds a token.

SD3.5 lives in a gated repo, so a token is required to fetch the checkpoint. Which token gets
used depends on HF_HOME, and a shell that never picked up the export -- an old terminal, tmux,
cron, a bare ``bash -c`` -- silently falls back to ``~/.cache/huggingface``. That yields a 401
if the fallback holds no token, or a 403 ("not in the authorized list") if it holds a different
account's, and neither message points at the real cause.

``huggingface_hub`` reads these paths once, at import time, so ``ensure_hf_home()`` has to run
before anything imports transformers/diffusers. Call it immediately after the ``sys.path``
setup and before the model imports; the call statement doubles as the barrier that stops isort
from reordering those imports above it.
"""

import os

# Searched in order when the current HF_HOME has no token. A directory qualifies if it holds a
# `token` file, which is where `hf auth login` writes the credential.
_CANDIDATE_HOMES = ("~/hf_data", "~/.cache/huggingface")


def _has_token(home):
    return bool(home) and os.path.exists(os.path.join(home, "token"))


def ensure_hf_home(verbose=True):
    """Set HF_HOME to a directory with a token, if the current setting lacks one.

    Precedence: ``SD35_HF_HOME`` (explicit override, used as-is) > an ``HF_HOME`` that already
    has a token > the first candidate that has one. Returns the resolved home, or ``None`` when
    no token was found anywhere -- in which case HF_HOME is left untouched and the caller will
    get the usual auth error from the hub.
    """
    explicit = os.environ.get("SD35_HF_HOME")
    if explicit:
        os.environ["HF_HOME"] = explicit
        return explicit

    current = os.environ.get("HF_HOME")
    if _has_token(current):
        return current

    for candidate in _CANDIDATE_HOMES:
        home = os.path.expanduser(candidate)
        if _has_token(home):
            if verbose and home != current:
                print(f"HF_HOME={current or '(unset)'} has no token; using {home}")
            os.environ["HF_HOME"] = home
            return home

    if verbose:
        print("warning: no Hugging Face token found; the gated SD3.5 repo will 401. Try `hf auth login`.")
    return None
