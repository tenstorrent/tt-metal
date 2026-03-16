# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Muon+AdamW optimizer.

Applies Muon (with Newton-Schulz orthogonalization) to 2D weight matrices
and AdamW to everything else (embeddings, norms, biases, LM head).

Registered as "MuonWithAdamW"
"""

import ttml

# Module names whose weights should use AdamW instead of Muon.
# Embeddings and LM head don't benefit from orthogonalization.
# Fused KV/QKV linears concatenate multiple matrices that should be
# orthogonalized separately.
# Splitting, applying step and concatenating is not yet supported for these parameters.
DEFAULT_ADAMW_MODULES = {"tok_emb", "pos_emb", "fc", "kv_linear", "qkv_linear"}


def _is_muon_param(name, adamw_modules):
    if not name.endswith("/weight"):
        return False
    return not any(part in adamw_modules for part in name.split("/"))


class MuonWithAdamW(ttml.optimizers.OptimizerBase):
    """
    Config YAML format:

        optimizer:
          type: MuonWithAdamW
          adamw_modules: [tok_emb, pos_emb, fc, kv_linear, qkv_linear]
          muon:
            lr: 0.02
            momentum: 0.95
            ns_steps: 5
          adamw:
            lr: 0.0003
            weight_decay: 0.01
    """

    def __init__(self, config, params):
        super().__init__(params)

        adamw_modules = set(config.get("adamw_modules", DEFAULT_ADAMW_MODULES))
        muon_cfg = config.get("muon", {})
        adamw_cfg = config.get("adamw", {})

        # Split parameters
        muon_params = {}
        adamw_params = {}

        for name, tensor in params.items():
            if _is_muon_param(name, adamw_modules):
                muon_params[name] = tensor
            else:
                adamw_params[name] = tensor

        # Create sub-optimizers
        self._muon = ttml.optimizers.MuonComposite(
            muon_params,
            ttml.optimizers.MuonConfig.make(
                lr=muon_cfg.get("lr", 0.02),
                momentum=muon_cfg.get("momentum", 0.95),
                ns_steps=muon_cfg.get("ns_steps", 5),
            ),
        )

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

    def set_state_dict(self, dict):
        self._muon.set_state_dict(dict["muon"])
        self._adamw.set_state_dict(dict["adamw"])

    def get_steps(self):
        return self._muon.get_steps()

    def set_steps(self, steps):
        self._muon.set_steps(steps)
        self._adamw.set_steps(steps)

    def get_lr(self):
        return self._muon.get_lr()

    def set_lr(self, lr):
        self._muon.set_lr(lr)

    def get_adamw_lr(self):
        return self._adamw.get_lr()

    def set_adamw_lr(self, lr):
        self._adamw.set_lr(lr)

    def print_param_groups(self):
        print(f"  Muon parameters ({len(self._muon_param_names)}):")
        for name in self._muon_param_names:
            print(f"    - {name}")
        print(f"  AdamW parameters ({len(self._adamw_param_names)}):")
        for name in self._adamw_param_names:
            print(f"    - {name}")


def register():
    ttml.optimizers.register_optimizer(
        "MuonWithAdamW", lambda config, params: MuonWithAdamW(config, params)
    )
