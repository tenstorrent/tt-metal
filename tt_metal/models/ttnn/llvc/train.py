#!/usr/bin/env python3
"""
LLVC (Low-Latency Low-Resource Voice Conversion) Training Script
Using TTNN APIs for model and training.

Usage:
    python train.py --epochs 100 --batch-size 32 --lr 0.001 --checkpoint-dir ./checkpoints
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader

import ttnn
from ttnn import Tensor, Module, optim, losses
from ttnn.optim import AdamW, SGD
from ttnn.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from ttnn.data import TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_INPUT_DIM = 80
DEFAULT_HIDDEN_DIM = 256
DEFAULT_OUTPUT_DIM = 80
DEFAULT_NUM_LAYERS = 3
DEFAULT_SEQ_LEN = 128
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_LOG_FILE = "llvc_training.log"
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 5

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure logging with console and optional file handler.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file.
        console: If True, add a console handler.

    Returns:
        Configured logger instance.

    Raises:
        ValueError: If the log level is invalid.
        OSError: If the log file cannot be opened.
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_upper = level.upper()
    if level_upper not in valid_levels:
        raise ValueError(f"Invalid log level '{level}'. Must be one of {valid_levels}")

    logger = logging.getLogger("LLVCTrainer")
    logger.setLevel(getattr(logging, level_upper))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_handler = logging.FileHandler(str(log_path))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning(f"Could not open log file {log_path}: {e}")

    return logger


logger = setup_logging()

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class LLVCTrainingError(Exception):
    """Base exception for LLVC training errors."""
    pass


class DeviceInitializationError(LLVCTrainingError):
    """Raised when TTNN device cannot be initialized."""
    pass


class DataLoadError(LLVCTrainingError):
    """Raised when data loading fails."""
    pass


class CheckpointError(LLVCTrainingError):
    """Raised when checkpoint save/load fails."""
    pass


class ConfigurationError(LLVCTrainingError):
    """Raised when configuration parameters are invalid."""
    pass


# ---------------------------------------------------------------------------
# Device management with error handling
# ---------------------------------------------------------------------------
def get_device(device_id: int = 0) -> ttnn.Device:
    """Initialize and return a TTNN device.

    Args:
        device_id: Device index to open (default: 0).

    Returns:
        Initialized TTNN device.

    Raises:
        DeviceInitializationError: If device cannot be opened.
    """
    try:
        device = ttnn.open_device(device_id=device_id)
        logger.info(f"Opened device: {device}")
        return device
    except Exception as e:
        logger.error(f"Failed to open device {device_id}: {e}")
        raise DeviceInitializationError(f"Device {device_id} open failed: {e}") from e


def close_device(device: ttnn.Device) -> None:
    """Safely close a TTNN device.

    Args:
        device: Device to close.
    """
    try:
        ttnn.close_device(device)
        logger.info("Device closed successfully.")
    except Exception as e:
        logger.warning(f"Error during device close: {e}")


# ---------------------------------------------------------------------------
# Dataset – with validation and error handling
# ---------------------------------------------------------------------------
class VoiceConversionDataset(Dataset):
    """
    Stub dataset for LLVC producing random mel-spectrograms for source and target.

    In production, replace with actual speech data loading and pre-processing.

    Args:
        num_samples: Number of samples in the dataset.
        input_dim: Dimension of input features (e.g., 80 mel bins).
        output_dim: Dimension of output features (e.g., 80 mel bins).
        seq_len: Number of time frames (default: 128).
        seed: Random seed for reproducibility (optional).

    Raises:
        ValueError: If any parameter is invalid.
    """

    def __init__(
        self,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        input_dim: int = DEFAULT_INPUT_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        seq_len: int = DEFAULT_SEQ_LEN,
        seed: Optional[int] = None,
    ) -> None:
        # Input validation
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

        if seed is not None:
            torch.manual_seed(seed)

        # Simulate data: source and target mel-spectrograms
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(input_dim, seq_len), torch.randn(output_dim, seq_len))
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data) - 1}]")
        return self.data[idx]


# ---------------------------------------------------------------------------
# Model – with type annotations, validation, and forward method
# ---------------------------------------------------------------------------
class VoiceConverter(ttnn.Module):
    """
    Low-latency voice conversion model using TTNN modules.

    A stack of linear layers with ReLU activations, followed by an output layer.
    The input is expected to be of shape (batch, input_dim, time).

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
        num_layers: Number of hidden layers (excluding output).
        dropout: Dropout probability (default 0.0, i.e., no dropout).

    Raises:
        ValueError: If any dimension parameter is non-positive or num_layers < 0.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if num_layers < 0:
            raise ValueError(f"num_layers cannot be negative, got {num_layers}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build layers
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(ttnn.Linear(prev_dim, hidden_dim))
            layers.append(ttnn.ReLU())
            if dropout > 0:
                layers.append(ttnn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(ttnn.Linear(prev_dim, output_dim))
        self.net = ttnn.Sequential(*layers)

        logger.info(
            f"VoiceConverter created: input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, num_layers={num_layers}, dropout={dropout}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim, time).

        Returns:
            Output tensor of shape (batch, output_dim, time).
        """
        # The network expects shape (batch, time, input_dim) for per-frame processing
        # We'll permute to (batch, time, input_dim) then back
        if x.dim() != 3:
            raise RuntimeError(f"Input must be 3D (batch, input_dim, time), got shape {x.shape}")
        # Permute: (batch, input_dim, time) -> (batch, time, input_dim)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        # Permute back: (batch, time, output_dim) -> (batch, output_dim, time)
        x = x.permute(0, 2, 1)
        return x


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    input_dim: int = DEFAULT_INPUT_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    output_dim: int = DEFAULT_OUTPUT_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = 0.0
    seq_len: int = DEFAULT_SEQ_LEN
    num_samples: int = DEFAULT_NUM_SAMPLES
    checkpoint_dir: Path = Path(DEFAULT_CHECKPOINT_DIR)
    resume: Optional[Path] = None
    log_file: Optional[Path] = Path(DEFAULT_LOG_FILE)
    device_id: int = 0


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    is_best: bool = False,
) -> None:
    """Save a checkpoint dictionary to disk.

    Args:
        state: Dictionary containing model state, optimizer, epoch, loss, etc.
        filepath: Path to save the checkpoint.
        is_best: If True, also save as 'best_model.pt' in the same directory.

    Raises:
        CheckpointError: If saving fails.
    """
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(filepath))
        logger.info(f"Checkpoint saved to {filepath}")
        if is_best:
            best_path = filepath.parent / "best_model.pt"
            # Use symlink or copy; copy is more portable
            torch.save(state, str(best_path))
            logger.info(f"Best model saved to {best_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {e}")
        raise CheckpointError(f"Save failed: {e}") from e


def load_checkpoint(
    filepath: Union[str, Path],
    model: ttnn.Module,
    optimizer: Optional[ttnn.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[ttnn.Device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint from disk and restore model/optimizer/scheduler state.

    Args:
        filepath: Path to the checkpoint file.
        model: The model to restore state into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Optional device to map tensors to.

    Returns:
        The checkpoint dictionary (contains 'epoch', 'best_loss', etc.).

    Raises:
        CheckpointError: If loading fails.
        FileNotFoundError: If checkpoint file not found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        checkpoint = torch.load(str(filepath), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state restored from checkpoint.")

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state restored.")

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state restored.")

        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {e}")
        raise CheckpointError(f"Load failed: {e}") from e


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def train_epoch(
    model: ttnn.Module,
    dataloader: DataLoader,
    loss_fn: ttnn.losses.Loss,
    optimizer: ttnn.optim.Optimizer,
    device: ttnn.Device,
    epoch: int,
    clip_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader providing input-target pairs.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: TTNN device.
        epoch: Current epoch number (for logging).
        clip_grad_norm: Max norm for gradient clipping (0 = no clipping).

    Returns:
        Average loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (source, target) in enumerate(dataloader):
        # Move data to device if needed
        source = source.to(device)
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(source)
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        total_loss += loss.item()

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{num_batches:4d} | "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(
    model: ttnn.Module,
    dataloader: DataLoader,
    loss_fn: ttnn.losses.Loss,
    device: ttnn.Device,
) -> float:
    """Evaluate the model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader with evaluation data.
        loss_fn: Loss function.
        device: TTNN device.

    Returns:
        Average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for source, target in dataloader:
            source = source.to(device)
            target = target.to(device)
            output = model(source)
            loss = loss_fn(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------
def run_training(config: TrainingConfig) -> None:
    """Main training loop with checkpointing, logging, and error handling.

    Args:
        config: Training configuration.

    Raises:
        ConfigurationError: If configuration is invalid.
        DeviceInitializationError: If device cannot be opened.
        CheckpointError: If checkpoint operations fail.
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    # Also set TTNN seed if available
    if hasattr(ttnn, 'manual_seed'):
        ttnn.manual_seed(config.seed)

    # Device initialization
    device = get_device(config.device_id)
    try:
        # Dataset and DataLoader
        logger.info("Creating dataset...")
        dataset = VoiceConversionDataset(
            num_samples=config.num_samples,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            seq_len=config.seq_len,
            seed=config.seed,
        )
        # Split into train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = data_utils.random_split(
            dataset, [train_size, val_size]
        )
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")

        # DataLoader with configured batch size, shuffle for train
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # For TTNN, usually 0 workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # Model
        model = VoiceConverter(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        model.to(device)

        # Loss function
        loss_fn = ttnn.losses.MSELoss()  # Change to appropriate loss for voice conversion

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: cosine annealing with warmup
        # For simplicity, use a cosine annealing scheduler without warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

        # Checkpoint resume logic
        start_epoch = 1
        best_loss = float('inf')
        if config.resume is not None:
            logger.info(f"Resuming from checkpoint: {config.resume}")
            checkpoint = load_checkpoint(
                config.resume,
                model,
                optimizer,
                scheduler,
                device,
            )
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")

        # Training loop
        logger.info("Starting training...")
        for epoch in range(start_epoch, config.epochs + 1):
            start_time = time.time()

            # Train one epoch
            train_loss = train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                epoch,
                clip_grad_norm=1.0,
            )

            # Step scheduler
            scheduler.step()

            # Validation
            val_loss = evaluate(model, val_loader, loss_fn, device)

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch:3d}/{config.epochs:3d} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Time: {elapsed:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            # Checkpoint saving
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
                'config': config,
            }

            # Save checkpoint every 10 epochs or if best
            if epoch % 10 == 0 or is_best:
                checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                save_checkpoint(checkpoint_state, checkpoint_path, is_best=is_best)

        # Save final model
        final_checkpoint_path = config.checkpoint_dir / "final_model.pt"
        final_state = {
            'epoch': config.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }
        save_checkpoint(final_state, final_checkpoint_path)
        logger.info("Training completed successfully.")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving interrupted checkpoint...")
        inter_checkpoint_path = config.checkpoint_dir / "interrupted.pt"
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
        }, inter_checkpoint_path)
        logger.info(f"Interrupted checkpoint saved to {inter_checkpoint_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        close_device(device)


# ---------------------------------------------------------------------------
# Argument parsing and configuration
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> TrainingConfig:
    """Parse command line arguments into a TrainingConfig.

    Args:
        argv: Argument list (default: sys.argv[1:]).

    Returns:
        TrainingConfig populated with parsed values.

    Raises:
        ConfigurationError: If argument validation fails.
    """
    parser = argparse.ArgumentParser(
        description="LLVC (Low-Latency Low-Resource Voice Conversion) Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    parser.add_argument('--input-dim', type=int, default=DEFAULT_INPUT_DIM,
                        help='Input feature dimension (mel bins)')
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM,
                        help='Hidden layer dimension')
    parser.add_argument('--output-dim', type=int, default=DEFAULT_OUTPUT_DIM,
                        help='Output feature dimension')
    parser.add_argument('--num-layers', type=int, default=DEFAULT_NUM_LAYERS,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability (0 = no dropout)')
    parser.add_argument('--seq-len', type=int, default=DEFAULT_SEQ_LEN,
                        help='Sequence length (time frames)')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help='Number of synthetic samples for training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', '--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_WARMUP_EPOCHS,
                        help='Number of warmup epochs (not implemented)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')

    # Checkpoint / logging
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log-file', type=Path, default=Path(DEFAULT_LOG_FILE),
                        help='Path to log file (empty for no file logging)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='Device ID for TTNN')

    # Parse
    args = parser.parse_args(argv)

    # Validation (more thorough than argparse constraints)
    if args.epochs <= 0:
        raise ConfigurationError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ConfigurationError(f"batch_size must be positive, got {args.batch_size}")
    if args.lr <= 0:
        raise ConfigurationError(f"learning_rate must be positive, got {args.lr}")
    if args.weight_decay < 0:
        raise ConfigurationError(f"weight_decay cannot be negative, got {args.weight_decay}")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ConfigurationError(f"dropout must be in [0, 1), got {args.dropout}")
    if args.seq_len <= 0:
        raise ConfigurationError(f"seq_len must be positive, got {args.seq_len}")
    if args.num_samples <= 0:
        raise ConfigurationError(f"num_samples must be positive, got {args.num_samples}")
    if args.input_dim <= 0:
        raise ConfigurationError(f"input_dim must be positive, got {args.input_dim}")
    if args.hidden_dim <= 0:
        raise ConfigurationError(f"hidden_dim must be positive, got {args.hidden_dim}")
    if args.output_dim <= 0:
        raise ConfigurationError(f"output_dim must be positive, got {args.output_dim}")
    if args.device_id < 0:
        raise ConfigurationError(f"device_id cannot be negative, got {args.device_id}")

    return TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        log_file=args.log_file,
        device_id=args.device_id,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for LLVC training.

    Sets up logging, parses arguments, and runs training.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).
    """
    try:
        config = parse_args(argv)

        # Reconfigure logging with file if specified
        if config.log_file:
            global logger
            logger = setup_logging(
                level="INFO",
                log_file=config.log_file,
                console=True,
            )

        logger.info("LLVC Training Script Started")
        logger.info(f"Configuration: {config}")

        run_training(config)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except DeviceInitializationError as e:
        logger.error(f"Device initialization error: {e}")
        sys.exit(2)
    except CheckpointError as e:
        logger.error(f"Checkpoint error: {e}")
        sys.exit(3)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(4)


if __name__ == "__main__":
    main()