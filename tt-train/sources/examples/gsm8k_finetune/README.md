## GSM8K Fine-tune Example (TT-Metal)

This example fine-tunes a TinyLlama-based causal LM on the GSM8K math word problems dataset using the TT-Train Python API for TTML, and provides a Streamlit dashboard for real-time monitoring and control.

### Directory
- `gsm8k_finetune.py`: Training script (runs fine-tuning and writes metrics)
- `streamlit_finetune_app.py`: Streamlit dashboard to configure, launch, and monitor training
- `slurm_training_service.py`: REST API service for tt-dashboard (SLURM job dispatch)
- `job_manager.py`: SLURM job submission and monitoring
- `openapi.yaml`: OpenAPI 3.0 specification for the training service API
- `run_dashboard.sh`: Helper script to install UI deps (if needed) and launch the dashboard
- `requirements_streamlit.txt`: Minimal requirements for the dashboard
- `requirements_service.txt`: Dependencies for the REST service (Flask, pyyaml)
### Prerequisites
- TT-Metal repo checked out and build/runtime set up
- Environment variable `TT_METAL_HOME` pointing to the repository root (e.g., `/home/ubuntu/tt-metal`)
- Python 3.10+ and a working internet connection (to download TinyLlama weights and GSM8K via Hugging Face)
- Tenstorrent hardware and tools available if you intend to use multi-device/TT-SMI features

Set the environment variable:

```bash
export TT_METAL_HOME=/home/ubuntu/tt-metal
```

### Base training config and overrides
The training script loads a base YAML config:
- Base: `tt-train/configs/training_shakespeare_tinyllama.yaml`

At runtime, if present, it applies overrides from:
- `tt-train/configs/training_overrides.yaml`

The dashboard can generate this overrides file for you with your selected hyperparameters and device mesh.

### Quick start: Launch the dashboard
From this directory:

```bash
cd /home/ubuntu/tt-metal/tt-train/sources/examples/gsm8k_finetune
./run_dashboard.sh
```

Then open `http://localhost:8501` in your browser.

If you prefer to run manually:

```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_finetune_app.py
```

### Using the dashboard
1. In the sidebar, choose:
   - Base model (e.g., `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`)
   - Devices/mesh (e.g., `N150`, `LoudBox`, `Galaxy`; enables DDP when mesh has >1 device)
   - Dataset (default: `gsm8k`)
   - Hyperparameters: learning rates, batch sizes, max steps, eval cadence, gradient accumulation, max sequence length
   - Optional: destination directory for `training_overrides.yaml` (defaults to `${TT_METAL_HOME}/tt-train/configs/`)
2. Click “Start Training” to:
   - Write `training_overrides.yaml`
   - Launch `python3 gsm8k_finetune.py` in this directory
3. Monitor progress:
   - Training Progress tab: current step, train/val loss, LR, plots
   - Validation Output tab: samples and last validation loss
   - Training Logs tab: tails of `training_stdout.log` and `training_stderr.log`
   - TT-SMI System Status: on-demand snapshot via `tt-smi -s`
4. Use “Stop Training” to gracefully terminate the process.

Notes:
- The dashboard reads metrics from `output.txt` and `validation.txt`, which the training script writes during execution.
- Enable Auto-refresh to update plots and metrics periodically.

### Training Service (REST API for tt-dashboard)

The training service exposes a REST API that bridges tt-dashboard to real TT hardware via SLURM. It accepts the dashboard request format, maps model/dataset/cluster IDs, and dispatches `sbatch` jobs.

**Run the service** (on a login node with SLURM access):

```bash
pip install -r requirements_service.txt
PORT=8085 JWT_SECRET=<same-as-dashboard> python slurm_training_service.py
```

**Jobs directory:** When `/data/$USER` exists, jobs are stored under
`/data/$USER/tt-metal/tt-train/sources/examples/gsm8k_finetune/jobs` so compute
nodes can write SLURM stdout/stderr. Override with `JOBS_BASE_DIR`.

Then point tt-dashboard at `http://<login-node>:8085` (`TT_TRAINING_SERVICE_URL` or `TT_TRAIN_BASE_URL`).

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/openapi.yaml` | OpenAPI 3.0 spec |
| GET | `/healthz` | Health check |
| GET | `/v1/catalog` | Capabilities and options (no auth) |
| GET | `/v1/jobs` | List jobs |
| POST | `/v1/jobs` | Create job |
| GET | `/v1/jobs/{id}` | Get job |
| POST | `/v1/jobs/{id}/cancel` | Cancel job |
| GET | `/v1/jobs/{id}/metrics` | Training metrics |
| GET | `/v1/jobs/{id}/logs` | Job logs |
| GET | `/v1/jobs/{id}/checkpoints` | Checkpoints |

**Auth:** `Authorization: Bearer <JWT>` when `JWT_SECRET` is set. Dev fallback: `X-TT-Organization` header.

See `GET /openapi.yaml` for the full schema; `GET /v1/catalog` for capabilities.

### OpenAPI Specification

The API is described by `openapi.yaml`. When the training service is running, the spec is served at:

```
GET http://<host>:8085/openapi.yaml
```

Use it for:
- API documentation and client generation
- Swagger UI / Redoc
- Validation tooling

The spec defines request/response schemas (CreateJobRequest, TrainingParams, OptimizerParams, Job, etc.) and documents which trainers/optimizers are supported. The dashboard should call `GET /v1/catalog` on load to get capabilities (`supported` flags).

### Running training directly (without the dashboard)
From this directory:

```bash
python3 gsm8k_finetune.py
```

What the script does:
- Loads tokenizer and downloads TinyLlama weights via Hugging Face Hub
- Loads GSM8K (`main` split) using `datasets`
- Builds a TTML Llama-style transformer according to the config
- Applies overrides from `${TT_METAL_HOME}/tt-train/configs/training_overrides.yaml` if that file exists
- Starts training and periodically validates; writes:
  - `output.txt` lines like: `LR: 3e-3, training_loss: 0.97, val_loss: 1.01, step: 120, epoch: 1`
  - `validation.txt` with sampled questions and generated answers, plus validation loss summary
- Saves `training_curves.png` at the end

### Configuration
- Base config: `tt-train/configs/training_shakespeare_tinyllama.yaml`
- Dashboard overrides produce YAML with sections:
  - `training_config` (e.g., `batch_size`, `max_steps`, `gradient_accumulation_steps`, `eval_every`, `transformer_config.max_sequence_length`)
  - `scheduler_config` (`min_lr`, `max_lr`, `warmup_steps`, `hold_steps`)
  - `device_config` (`enable_ddp`, `mesh_shape`)

Multi-device: If `device_config.total_devices() > 1`, the script initializes device(s) and synchronizes parameters between steps. Choose an appropriate `mesh_shape` for your system.

### Outputs and artifacts
- `output.txt`: training step metrics (parsed live by the dashboard)
- `validation.txt`: validation samples and last validation loss
- `training_curves.png`: loss curves at the end of training
- `training_stdout.log`, `training_stderr.log`: process logs (dashboard shows tails)

### Troubleshooting
- Dashboard does not start: ensure `pip install -r requirements_streamlit.txt` and that `streamlit` is on your PATH
- No metrics in UI: verify `output.txt` is being written; check `training_stdout.log` and `training_stderr.log`
- `tt-smi` not found: install Tenstorrent tools and ensure they’re on PATH
- HF downloads blocked: ensure internet access or pre-cache the TinyLlama model and datasets
- DDP/multi-device: ensure your `mesh_shape` matches available devices; adjust batch sizes and accumulation if you hit memory limits
- **Training service 401:** set `JWT_SECRET` to match tt-dashboard, or use `X-TT-Organization` header when `JWT_SECRET` is unset
- **Training service 400 (unsupported_trainer/optimizer):** check `GET /v1/catalog` for supported options; currently only trainer=sft and optimizer=adamw are supported

### License/attribution
- GSM8K: by Cobbe et al., distributed via Hugging Face `datasets`
- TinyLlama weights: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` via Hugging Face Hub
