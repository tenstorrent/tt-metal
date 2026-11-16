# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Muon+AdamW optimizer.

Applies Muon (with Newton-Schulz orthogonalization) to 2D weight matrices
and AdamW to everything else (embeddings, norms, biases, LM head).

Registered as "MuonWithAdamW"
"""

import ttml

# Module names whose weights should use AdamW instead of Muon.
# Embeddings and LM head don't benefit from orthogonalization.
# Fused linears concatenate multiple matrices that should be orthogonalized
# separately: qkv_linear ([Q|K|V], NanoGPT and Llama), w_gate_up ([gate|up], Llama)
# and kv_linear ([K|V], the C++ Llama).
# TODO(https://github.com/tenstorrent/tt-metal/issues/60893): apply Muon per block,
# rather than sending them to AdamW.
DEFAULT_ADAMW_MODULES = {"tok_emb", "pos_emb", "fc", "kv_linear", "qkv_linear", "w_gate_up"}


def _is_muon_param(name, adamw_modules):
    if not name.endswith("/weight"):
        return False
    return not any(part in adamw_modules for part in name.split("/"))


class MuonWithAdamW(ttml.optimizers.OptimizerBase):
    """
    Config YAML format:

        optimizer:
          type: MuonWithAdamW
          adamw_modules: [tok_emb, pos_emb, fc, kv_linear, qkv_linear, w_gate_up]
          muon:
            lr: 0.02
            momentum: 0.95
            ns_steps: 5
          adamw:
            lr: 0.0003
            weight_decay: 0.01

    Param groups: the Muon optimizer's groups followed by the AdamW optimizer's
    (``get_param_groups``, ``get_group_lrs``, ``set_group_lrs`` and
    ``get_initial_group_lrs`` all use that order). So one LR scheduler attached
    to the wrapper schedules every group of both, each from its own base LR,
    like a PyTorch scheduler over one optimizer with several param groups:

        opt = MuonWithAdamW(config, params)
        sched = CosineAnnealingScheduler(opt, T_max=1000)

    To give Muon and AdamW different schedules, attach one scheduler to
    ``muon_optimizer()`` and another to ``adamw_optimizer()`` instead.

    Each inner optimizer records and checkpoints its own ``initial_lr``
    (nested in this wrapper's state dict), so schedulers re-attached after a
    resume see the correct base LRs.
    """

    def __init__(self, config, params):
        super().__init__(params)

        adamw_modules = set(config.get("adamw_modules", DEFAULT_ADAMW_MODULES))
        muon_cfg = config.get("muon", {})
        adamw_cfg = config.get("adamw", {})
        # Set at the top level by trainers under pipeline parallelism; the groups it applies to live in the inner
        # sections, so pass it down to both.
        if config.get("allow_empty_param_groups"):
            muon_cfg = {**muon_cfg, "allow_empty_param_groups": True}
            adamw_cfg = {**adamw_cfg, "allow_empty_param_groups": True}

        # Split parameters
        muon_params = {}
        adamw_params = {}

        for name, tensor in params.items():
            if _is_muon_param(name, adamw_modules):
                muon_params[name] = tensor
            else:
                adamw_params[name] = tensor

        # TODO: add FSDP-aware Muon.
        muon_dict = {"type": "MuonComposite", "lr": 0.02, "momentum": 0.95, "ns_steps": 5, **muon_cfg}
        self._muon = ttml.optimizers.create_optimizer(muon_dict, muon_params)

        adamw_dict = {"type": "AdamW", **adamw_cfg}
        self._adamw = ttml.optimizers.create_optimizer(adamw_dict, adamw_params)

        self._muon_param_names = sorted(muon_params.keys())
        self._adamw_param_names = sorted(adamw_params.keys())

    def get_name(self):
        return "MuonWithAdamW"

    def zero_grad(self):
        self._muon.zero_grad()
        self._adamw.zero_grad()

    def step(self):
        self._muon.step()
        self._adamw.step()

    def get_state_dict(self):
        return {
            "muon": self._muon.get_state_dict(),
            "adamw": self._adamw.get_state_dict(),
        }

    def set_state_dict(self, state_dict):
        self._muon.set_state_dict(state_dict["muon"])
        self._adamw.set_state_dict(state_dict["adamw"])

    def get_steps(self):
        return self._muon.get_steps()

    def set_steps(self, steps):
        self._muon.set_steps(steps)
        self._adamw.set_steps(steps)

    def muon_optimizer(self):
        """Inner Muon optimizer (2D hidden-layer weights). Attach its LR scheduler here."""
        return self._muon

    def adamw_optimizer(self):
        """Inner AdamW optimizer (embeddings, norms, biases, LM head). Attach its LR scheduler here."""
        return self._adamw

    def get_param_groups(self):
        return self._muon.get_param_groups() + self._adamw.get_param_groups()

    def get_group_options(self, key):
        # Raises for an option one of the inner optimizers lacks (Muon has no weight_decay, AdamW no momentum).
        return self._muon.get_group_options(key) + self._adamw.get_group_options(key)

    def set_group_options(self, key, values):
        num_muon_groups = len(self._muon.get_param_groups())
        num_groups = num_muon_groups + len(self._adamw.get_param_groups())
        if len(values) != num_groups:
            raise ValueError(f"MuonWithAdamW has {num_groups} param groups, but got {len(values)} values for '{key}'")
        # Pre-check only: raises if either inner optimizer lacks `key`, before Muon is changed and AdamW then rejects it.
        self.get_group_options(key)
        self._muon.set_group_options(key, list(values[:num_muon_groups]))
        self._adamw.set_group_options(key, list(values[num_muon_groups:]))

    def set_param_group_option(self, group_idx, key, value):
        num_muon_groups = len(self._muon.get_param_groups())
        if group_idx < num_muon_groups:
            self._muon.set_param_group_option(group_idx, key, value)
        else:
            self._adamw.set_param_group_option(group_idx - num_muon_groups, key, value)

    def get_group_lrs(self):
        return self.get_group_options("lr")

    def set_group_lrs(self, lrs):
        self.set_group_options("lr", lrs)

    def get_initial_group_lrs(self):
        return self._muon.get_initial_group_lrs() + self._adamw.get_initial_group_lrs()

    def print_param_groups(self):
        print(f"  Muon parameters ({len(self._muon_param_names)}):")
        for name in self._muon_param_names:
            print(f"    - {name}")
        print(f"  AdamW parameters ({len(self._adamw_param_names)}):")
        for name in self._adamw_param_names:
            print(f"    - {name}")


def register():
    ttml.optimizers.register_optimizer("MuonWithAdamW", MuonWithAdamW)
