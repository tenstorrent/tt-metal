# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the generation stop-token list from the checkpoint's own eos ids."""


def _as_id_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value if v is not None]
    return [int(value)]


def build_stop_tokens(tokenizer_eos, config_eos=None, generation_eos=None, extra=()):
    """Return a deduplicated stop list: tokenizer eos first, then the config and
    generation_config eos ids (int or list), then any extra ids, in that order."""
    out = []
    for tok in _as_id_list(tokenizer_eos) + _as_id_list(config_eos) + _as_id_list(generation_eos) + _as_id_list(extra):
        if tok not in out:
            out.append(tok)
    return out
