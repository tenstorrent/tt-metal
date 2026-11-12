# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, ClassVar, Literal, Optional

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny


class OptimizerConfig(BaseModel):
    """What every optimizer takes. Each optimizer subclasses it with its own fields, named as ttml names them."""

    model_config = ConfigDict(extra="forbid")

    # The ttml optimizer type a subclass builds.
    ttml_type: ClassVar[str]

    # Constant over the run.
    lr: float = Field(ge=0.0)
    # ttml module names: the trainable weights with one of them as a `/`-separated part of their name, as ttml's
    # param groups match. Unset trains every trainable weight no other optimizer of the run trains.
    modules: Optional[list[str]] = Field(default=None, min_length=1)

    def to_ttml(self) -> dict:
        """The config ``ttml.optimizers.create_optimizer`` takes; unset fields keep ttml's defaults."""
        return {**self.model_dump(exclude={"type", "modules"}, exclude_none=True), "type": self.ttml_type}


class AdamWConfig(OptimizerConfig):
    ttml_type: ClassVar[str] = "AdamW"

    type: Literal["adamw"] = "adamw"
    # Decoupled weight decay.
    weight_decay: float = Field(default=0.0, ge=0.0)
    beta1: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    beta2: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    epsilon: Optional[float] = Field(default=None, gt=0.0)
    amsgrad: Optional[bool] = None
    # For bf16 parameters.
    stochastic_rounding: Optional[bool] = None


# The optimizers a config can name in `type`.
OPTIMIZERS: dict[str, type[OptimizerConfig]] = {"adamw": AdamWConfig}


def _named_optimizer(data: Any) -> Any:
    """Validate a mapping as the optimizer its ``type`` names, ``adamw`` if unset."""
    if isinstance(data, OptimizerConfig) and type(data) not in OPTIMIZERS.values():
        raise ValueError(f"{type(data).__name__} is not one of the optimizers {sorted(OPTIMIZERS)}")
    if not isinstance(data, dict):
        return data
    name = data.get("type", "adamw")
    if name not in OPTIMIZERS:
        raise ValueError(f"type: {name!r} is not one of {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name].model_validate(data)


def _as_list(data: Any) -> Any:
    return [data] if isinstance(data, (dict, OptimizerConfig)) else data


def _one_takes_the_rest(optimizers: list[OptimizerConfig]) -> list[OptimizerConfig]:
    if sum(optimizer.modules is None for optimizer in optimizers) > 1:
        raise ValueError("at most one optimizer may leave modules unset to train the remaining weights")
    return optimizers


# One optimizer, or a list of them over disjoint weights. Serialized with each optimizer's own fields.
OptimizerConfigs = Annotated[
    list[SerializeAsAny[Annotated[OptimizerConfig, BeforeValidator(_named_optimizer)]]],
    BeforeValidator(_as_list),
    AfterValidator(_one_takes_the_rest),
    Field(min_length=1),
]
