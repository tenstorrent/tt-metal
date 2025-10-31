# Hierarchical Parallel Training

This directory contains a Python implementation of hierarchical parallel training with support for both 2-tier and 3-tier architectures.

## Architecture Overview

The training process can be configured in two different modes:

### 2-Tier Architecture (Workers + AggregatorOptimizer)

The 2-tier architecture combines gradient aggregation and optimization into a single host, reducing communication overhead:

```
┌─────────────────────────────────────────────────────────────┐
│                    2-Tier Architecture                       │
└─────────────────────────────────────────────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 0 │  │ Worker 1 │  │ Worker N │  ← Compute forward/backward
└─────┬────┘  └─────┬────┘  └─────┬────┘    Send gradients
      │             │              │
      └─────────────┴──────────────┘
                    │ Gradients
                    ▼
        ┌───────────────────────────┐
        │  AggregatorOptimizer      │      ← Average gradients
        │      (Rank N)             │        Apply optimizer step
        └───────────┬───────────────┘        Send updated weights
                    │ Updated Weights
      ┌─────────────┴──────────────┐
      │             │              │
      ▼             ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 0 │  │ Worker 1 │  │ Worker N │  ← Receive updated weights
└──────────┘  └──────────┘  └──────────┘
```

**Total ranks needed**: `num_workers + 1`

### 3-Tier Architecture (Workers + Aggregator + Optimizer)

The 3-tier architecture separates gradient aggregation and optimization into separate hosts:

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

**Total ranks needed**: `num_workers + 2`

## Components

### Common Components (Both Architectures)

**Workers (Ranks 0 to N-1)**:
- Compute forward and backward passes
- Use `RemoteOptimizer` to communicate with aggregator/aggregator_optimizer
- Send gradients after backward pass
- Receive updated weights

### 2-Tier Specific Components

**AggregatorOptimizer (Rank N)**:
- Receives gradients from all workers
- Averages gradients across workers
- Optionally applies DDP reduction (cross-device averaging)
- Sets gradients on model parameters using `set_grad_from_tensor()`
- Applies optimizer step (e.g., AdamW with momentum, weight decay)
- Broadcasts updated weights to all workers

### 3-Tier Specific Components

**Aggregator (Rank N)**:
- Receives gradients from all workers
- Averages gradients across workers
- Optionally applies DDP reduction (cross-device averaging)
- Sends averaged gradients to optimizer
- Receives updated weights from optimizer
- Broadcasts weights to all workers

**Optimizer (Rank N+1)**:
- Receives averaged gradients from aggregator
- Applies optimizer step (e.g., AdamW with momentum, weight decay)
- Sends updated weights back to aggregator

## Files

- `training.py` - Main entry point that automatically detects architecture mode and dispatches to appropriate worker type
- `trainer.py` - **All worker implementations in one file:**
  - `worker()` - Worker training loop using RemoteOptimizer (both architectures)
  - `aggregator_optimizer()` - Combined aggregation and optimization (2-tier only)
  - `aggregator()` - Aggregates gradients from workers (3-tier only)
  - `optimizer()` - Applies optimizer updates (3-tier only)
- `data.py` - Data loading and preparation
- `runner.sh` - Script to launch multi-host training

## Usage

The `training.py` script **automatically detects** which architecture to use based on `world_size` and `num_workers`:

- **2-tier mode**: `world_size == num_workers + 1`
- **3-tier mode**: `world_size == num_workers + 2`

### 2-Tier Example

```bash
# For num_workers=4, you need 5 total ranks: 4 workers + 1 aggregator_optimizer
# Launch with 5 MPI ranks
./runner.sh --config config_with_4_workers.yaml
```

The script automatically determines worker type based on rank:
- Ranks 0-2: Training workers → calls `worker()` from `trainer.py`
- Rank 3: AggregatorOptimizer → calls `aggregator_optimizer()` from `trainer.py`

### 3-Tier Example

```bash
# For num_workers=3, you need 5 total ranks: 3 workers + 1 aggregator + 1 optimizer
# Launch with 5 MPI ranks
./runner.sh --config config_with_3_workers.yaml
```

The script automatically determines worker type based on rank:
- Ranks 0-2: Training workers → calls `worker()` from `trainer.py`
- Rank 3: Aggregator → calls `aggregator()` from `trainer.py`
- Rank 4: Optimizer → calls `optimizer()` from `trainer.py`


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

### 2-Tier Architecture (Per Training Step):

1. **Workers → AggregatorOptimizer**: Each worker sends gradients
2. **AggregatorOptimizer**:
   - Averages gradients across workers
   - Applies DDP if enabled
   - Sets gradients on model parameters
   - Applies optimizer step (e.g., AdamW)
3. **AggregatorOptimizer → Workers**: Broadcasts updated weights to all workers

### 3-Tier Architecture (Per Training Step):

1. **Workers → Aggregator**: Each worker sends gradients
2. **Aggregator**: Averages gradients, applies DDP if enabled
3. **Aggregator → Optimizer**: Sends averaged gradients
4. **Optimizer**: Applies optimizer step (e.g., AdamW)
5. **Optimizer → Aggregator**: Sends updated weights
6. **Aggregator → Workers**: Broadcasts updated weights to all workers


## RemoteOptimizer

The `RemoteOptimizer` class provides a simple interface for workers in the `worker()` function:

```python
# Create remote optimizer pointing to aggregator (or aggregator_optimizer in 2-tier)
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
