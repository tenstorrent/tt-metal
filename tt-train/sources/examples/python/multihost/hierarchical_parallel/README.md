# 3-Tier Architecture Training

This directory contains a Python implementation of the 3-tier architecture training

## Architecture Overview

The 3-tier architecture separates the training process into three types of workers:

```
┌─────────────────────────────────────────────────────────────┐
│                    3-Tier Architecture                       │
└─────────────────────────────────────────────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 0 │  │ Worker 1 │  │ Worker N │  ← Compute forward/backward
└─────┬────┘  └─────┬────┘  └─────┬────┘    Send gradients
      │             │              │
      └─────────────┴──────────────┘
                    │ Gradients
                    ▼
            ┌───────────────┐
            │  Aggregator   │              ← Average gradients
            │ (Rank N+1)    │                Apply DDP if enabled
            └───────┬───────┘                Send to optimizer
                    │ Avg Gradients
                    ▼
            ┌───────────────┐
            │  Optimizer    │              ← Apply optimizer step
            │ (Rank N+2)    │                (AdamW, SGD, etc.)
            └───────┬───────┘
                    │ Updated Weights
                    ▼
            ┌───────────────┐
            │  Aggregator   │              ← Broadcast weights
            └───────┬───────┘
                    │
      ┌─────────────┴──────────────┐
      │             │              │
      ▼             ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 0 │  │ Worker 1 │  │ Worker N │  ← Receive updated weights
└──────────┘  └──────────┘  └──────────┘
```

### Components

1. **Workers (Ranks 0 to N-1)**:
   - Compute forward and backward passes
   - Use `RemoteOptimizer` to communicate with aggregator
   - Send gradients to aggregator after backward pass
   - Receive updated weights from aggregator

2. **Aggregator (Rank N)**:
   - Receives gradients from all workers
   - Averages gradients across workers
   - Optionally applies DDP reduction (cross-device averaging)
   - Sends averaged gradients to optimizer
   - Receives updated weights from optimizer
   - Broadcasts weights to all workers

3. **Optimizer (Rank N+1)**:
   - Receives averaged gradients from aggregator
   - Applies optimizer step (e.g., AdamW with momentum, weight decay)
   - Sends updated weights back to aggregator

## Files

- `training.py` - Main entry point that dispatches to appropriate worker type
- `trainer.py` - **All worker implementations in one file:**
  - `worker()` - Worker training loop using RemoteOptimizer
  - `aggregator()` - Aggregates gradients from workers
  - `optimizer()` - Applies optimizer updates
- `data.py` - Data loading and preparation
- `runner.sh` - Script to launch multi-host training

## Usage

Use `training.py` which automatically dispatches to the correct worker type:

```bash
# For num_workers=2, you need 4 total ranks: 2 workers + 1 aggregator + 1 optimizer
./runner.sh
```

The script automatically determines worker type based on rank:
- Ranks 0 to num_workers-1: Training workers → calls `worker()` from `trainer.py`
- Rank num_workers: Aggregator → calls `aggregator()` from `trainer.py`
- Rank num_workers+1: Optimizer → calls `optimizer()` from `trainer.py`

All three worker implementations are in `trainer.py`, making it easy to see what each type of host does.


## Configuration

The configuration file should include:

```yaml
training_config:
  steps: 1000
  batch_size: 32
  gradient_accumulation_steps: 1
  num_epochs: 1

multihost_config:
  enabled: true
  num_workers: 2  # Number of training worker ranks
  socket_type: fabric  # or mpi

device_config:
  enable_ddp: true  # Enable if using multiple devices per worker
  enable_tp: false   # Enable for tensor parallelism
  mesh_shape: [1, 8]
```

## Communication Pattern

### Per Training Step:

1. **Workers → Aggregator**: Each worker sends gradients
2. **Aggregator**: Averages gradients, applies DDP if enabled
3. **Aggregator → Optimizer**: Sends averaged gradients
4. **Optimizer**: Applies optimizer step (e.g., AdamW)
5. **Optimizer → Aggregator**: Sends updated weights
6. **Aggregator → Workers**: Broadcasts updated weights to all workers


## RemoteOptimizer

The `RemoteOptimizer` class provides a simple interface for workers in the `worker()` function:

```python
# Create remote optimizer pointing to aggregator
remote_opt = ttml.optimizers.RemoteOptimizer(model.parameters(), aggregator_rank)

# Receive initial weights
remote_opt.receive_weights()

# Training loop
for step in range(num_steps):
    remote_opt.zero_grad()

    # Forward and backward
    loss = model(x, y)
    loss.backward()

    # Send gradients and receive updated weights
    remote_opt.step()  # Handles all communication internally
```
