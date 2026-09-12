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

    LR scheduling contract: this optimizer holds TWO learning rates, so LR
    schedulers must be attached to the inner optimizers individually — one to
    ``muon_optimizer()`` and one to ``adamw_optimizer()`` (mirroring PyTorch's
    idiom of a separate ``torch.optim.Muon`` and ``torch.optim.AdamW``, each
    with its own scheduler). Attaching a scheduler to the wrapper itself
    raises. Example:

        opt = MuonWithAdamW(config, params)
        muon_sched = CosineAnnealingScheduler(opt.muon_optimizer(), T_max=1000)
        adamw_sched = CosineAnnealingScheduler(opt.adamw_optimizer(), T_max=1000)
        ...
        opt.step()
        muon_sched.step()
        adamw_sched.step()

    Each inner optimizer records and checkpoints its own ``initial_lr``
    (nested in this wrapper's state dict), so schedulers re-attached after a
    resume see the correct per-optimizer base LRs.
    """

    # TODO(#tt-metal issue pending): what contract should get_lr/set_lr (and
    # get_adamw_lr/set_adamw_lr) honor on this fused optimizer? Today they
    # address only one of the two internal LRs, which is why schedulers are
    # rejected on the wrapper (see supports_lr_scheduling) — but the same
    # hazard exists for any generic code that calls set_lr directly (e.g.
    # grpo_trainer's manual warmup) or logs get_lr. Candidates: raise on both,
    # or make set_lr scale both LRs proportionally from their bases.
    supports_lr_scheduling = False
    lr_scheduling_hint = "Attach one scheduler to .muon_optimizer() and another to .adamw_optimizer() instead."

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

        # TODO: add FSDP-aware Muon.
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
    ttml.optimizers.register_optimizer("MuonWithAdamW", MuonWithAdamW)
