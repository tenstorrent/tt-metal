import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import yaml
import ttml
import click
import random
import numpy as np
from data import prepare_data, get_batch, build_causal_mask

from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    ttml.autograd.AutoContext.get_instance().set_seed(seed)


def get_config(path: str):
    path = f'{os.environ["TT_METAL_HOME"]}/tt-train/configs/{path}'
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


class DeviceConfig:
    def __init__(self, yaml_config):
        device_config = yaml_config.get("device_config", {})
        self.mesh_shape = device_config.get("mesh_shape", [1, 1])
        self.device_ids = device_config.get("device_ids", [])
        self.enable_tp = device_config.get("enable_tp", False)
        self.enable_ddp = device_config.get("enable_ddp", False)

        # we currently support only [1, N] mesh shapes
        assert self.mesh_shape[0] == 1

    def total_devices(self):
        return self.mesh_shape[0] * self.mesh_shape[1]


def initialize_device(yaml_config):
    device_config = DeviceConfig(yaml_config)
    ttml.core.distributed.enable_fabric(device_config.total_devices())
    ttml.autograd.AutoContext.get_instance().open_device(device_config.mesh_shape, device_config.device_ids)


class ModelConfig:
    def __init__(self, yaml_config):
        self.device_config = DeviceConfig(yaml_config)
        self.model_type = yaml_config.get("model_type", "gpt2")
        self.transformer_config = yaml_config.get("transformer_config", {})

    def _create_gpt2(self):
        gcfg = ttml.models.gpt2.GPT2TransformerConfig()
        gcfg.num_heads = self.transformer_config.get("num_heads", 6)
        gcfg.embedding_dim = self.transformer_config.get("embedding_dim", 384)
        gcfg.num_blocks = self.transformer_config.get("num_blocks", 6)
        gcfg.vocab_size = self.transformer_config.get("vocab_size", 256)
        gcfg.max_sequence_length = self.transformer_config.get("max_sequence_length", 256)
        gcfg.dropout_prob = self.transformer_config.get("dropout_prob", 0.2)
        return ttml.models.gpt2.create_gpt2_model(gcfg)

    def _create_llama(self):
        raise NotImplementedError("LLama model is not supported yet")
        # lcfg = ttml.models.llama.LlamaConfig()
        # lcfg.num_heads = self.transformer_config.get("num_heads", 6)
        # lcfg.num_groups = self.transformer_config.get("num_groups", 3)
        # lcfg.embedding_dim = self.transformer_config.get("embedding_dim", 384)
        # lcfg.num_blocks = self.transformer_config.get("num_blocks", 6)
        # lcfg.vocab_size = self.transformer_config.get("vocab_size", 256)
        # lcfg.max_sequence_length = self.transformer_config.get("max_sequence_length", 256)
        # lcfg.dropout_prob = self.transformer_config.get("dropout_prob", 0.2)
        # return ttml.models.llama.create(lcfg)

    def create_model(self):
        if self.model_type == "gpt2":
            return self._create_gpt2()
        elif self.model_type == "llama":
            return self._create_llama()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")


class TrainingConfig:
    def __init__(self, yaml_config):
        tc = yaml_config.get("training_config", {})
        self.batch_size = int(tc.get("batch_size", 64))
        self.steps = int(tc.get("max_steps", 1000))
        self.eval_every = int(tc.get("eval_every", 200))
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))

        tcfg = tc.get("transformer_config", yaml_config.get("transformer_config", {}))
        self.seq_len = int(tcfg.get("max_sequence_length", 256))


def create_optimizer(model, yaml_config):
    lr = yaml_config.get("learning_rate", 0.0003)
    beta1 = yaml_config.get("beta1", 0.9)
    beta2 = yaml_config.get("beta2", 0.999)
    eps = yaml_config.get("eps", 1e-8)
    weight_decay = yaml_config.get("weight_decay", 0.01)

    adamw_cfg = ttml.optimizers.AdamWConfig.make(
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
    )
    return ttml.optimizers.AdamW(model.parameters(), adamw_cfg)


def get_batch_ttml(ids, seq_len, batch_size, use_ddp=False):
    device = ttml.autograd.AutoContext.get_instance().get_device()
    x_u32, y_u32 = get_batch(ids, seq_len, batch_size)
    # TTML shapes: inputs [B,1,1,T] (uint32), targets [B,T] (int32)

    if use_ddp:
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
        tt_x = ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(batch_size, 1, 1, seq_len), ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32, mapper
        )
        tt_y = ttml.autograd.Tensor.from_numpy(y_u32, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32, mapper)
    else:
        tt_x = ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(batch_size, 1, 1, seq_len), ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
        )
        tt_y = ttml.autograd.Tensor.from_numpy(y_u32, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)
    return tt_x, tt_y


def train(cfg, model, optim, train_ids: np.ndarray, val_ids: np.ndarray, use_ddp=False):
    loss_fn = ttml.ops.loss.cross_entropy_loss
    reduce = ttml.ops.ReduceType.MEAN

    causal_mask = build_causal_mask(cfg.seq_len)
    tt_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )  # [1,1,T,T], float32

    # Create composer for distributed tensors if using DDP
    composer = None
    if use_ddp:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    model.train()
    train_losses = []
    val_losses = []

    # Gradient accumulation state
    accum_step = 0
    accum_loss = 0.0
    last_val_loss = None

    bar = tqdm(range(1, cfg.steps + 1))
    for step in bar:
        tt_x, tt_y = get_batch_ttml(train_ids, cfg.seq_len, cfg.batch_size, use_ddp)

        # Zero grad only at the start of accumulation
        if accum_step == 0:
            optim.zero_grad()

        # ---- forward/backward ----
        logits = model(tt_x, tt_mask)
        loss = loss_fn(logits, tt_y, reduce)

        # Scale loss by accumulation steps
        if cfg.gradient_accumulation_steps > 1:
            loss = ttml.ops.mul(loss, 1.0 / cfg.gradient_accumulation_steps)

        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # For DDP, composer concatenates losses from all devices - take mean
        loss_numpy = loss.to_numpy(composer=composer)
        train_loss = float(loss_numpy.mean() if use_ddp else loss_numpy)

        # Accumulate loss
        accum_loss += train_loss
        accum_step += 1

        # Step optimizer when accumulation is complete
        if accum_step >= cfg.gradient_accumulation_steps:
            # Synchronize gradients for DDP
            if use_ddp:
                ttml.core.distributed.synchronize_parameters(model.parameters())

            optim.step()

            # Record average accumulated loss
            avg_loss = accum_loss / cfg.gradient_accumulation_steps
            train_losses.append(avg_loss)

            # Update progress bar - preserve val_loss if it exists
            postfix = {"train_loss": f"{avg_loss:.4f}"}
            if last_val_loss is not None:
                postfix["val_loss"] = f"{last_val_loss:.4f}"
            bar.set_postfix(postfix, refresh=False)

            # Reset accumulation
            accum_step = 0
            accum_loss = 0.0

        # ---- occasional eval on val set ----
        if (step % cfg.eval_every) == 0 or step == 1:
            model.eval()
            # keep existing placeholder behavior for validation loss
            val_losses.append(train_losses[-1] if train_losses else 0.0)
            last_val_loss = val_losses[-1]
            model.train()
            # Update bar with validation loss
            postfix = {"train_loss": f"{train_losses[-1]:.4f}" if train_losses else "N/A"}
            postfix["val_loss"] = f"{last_val_loss:.4f}"
            bar.set_postfix(postfix, refresh=False)

    return train_losses, val_losses


@click.command()
@click.option("-c", "--config", type=str, default="training_shakespeare_nanogpt.yaml")
def main(config: str):
    set_seed(42)
    yaml_config = get_config(config)
    train_ids, val_ids, vocab_size, decode = prepare_data()

    initialize_device(yaml_config)
    device = ttml.autograd.AutoContext.get_instance().get_device()
    print(device)

    model_config = ModelConfig(yaml_config)
    model = model_config.create_model()
    print(model)

    optimizer = create_optimizer(model, yaml_config)
    print(optimizer)

    training_cfg = TrainingConfig(yaml_config)
    device_config = DeviceConfig(yaml_config)
    train_losses, val_losses = train(training_cfg, model, optimizer, train_ids, val_ids, device_config.enable_ddp)

    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
