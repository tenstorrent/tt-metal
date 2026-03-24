#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SLURM Job Manager for GSM8K Fine-tuning Dashboard
Handles job submission, monitoring, and management via SLURM.
"""

import json
import os
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

from ttml.common.utils import get_tt_metal_runtime_root


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInfo:
    """Information about a submitted SLURM job."""

    job_id: str
    job_name: str
    partition: str
    nodes: int
    status: str
    submit_time: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    config: Dict = field(default_factory=dict)
    output_dir: str = ""
    training_history: List[Dict] = field(default_factory=list)
    last_step: int = -1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "JobInfo":
        return cls(**data)


# Default partition to device mesh shape mapping
PARTITION_DEVICE_MAPPING = {
    "bh_lb_single": {
        "mesh_shape": [8, 1],
        "description": "LoudBox Single Node (8x1)",
        "max_nodes": 1,
    },
    "bh_single": {
        "mesh_shape": [32, 1],
        "description": "BH BLX",
        "max_nodes": 1,
    },
    "bh_pod_4x32_B45": {
        "mesh_shape": [32, 1],
        "description": "4 BH GLX",
        "max_nodes": 4,
    },
    "bh_pod_4x32_B89": {
        "mesh_shape": [32, 1],
        "description": "4 BH GLX",
        "max_nodes": 4,
    },
    "bh_sp_5x4x32_C1_C10": {
        "mesh_shape": [32, 1],
        "description": "5 x 4 BH GLX",
        "max_nodes": 20,
    },
    "bh_lb_multi": {
        "mesh_shape": [8, 1],
        "description": "LoudBox Multi Node",
        "max_nodes": 4,
    },
    "bh_galaxy": {
        "mesh_shape": [32, 1],
        "description": "Galaxy (32x1)",
        "max_nodes": 1,
    },
    "n150": {
        "mesh_shape": [1, 1],
        "description": "N150 Single Device (1x1)",
        "max_nodes": 1,
    },
    "n300": {
        "mesh_shape": [1, 2],
        "description": "N300 Dual Device (1x2)",
        "max_nodes": 1,
    },
}

# Available Galaxy nodes for non-lb partitions
GALAXY_NODES = [
    "bh-glx-c01u02",
    "bh-glx-c01u08",
    "bh-glx-c02u02",
    "bh-glx-c02u08",
]

# Mesh graph descriptor template for LoudBox (lb) partitions
LB_MESH_GRAPH_TEMPLATE = """# --- Meshes ---------------------------------------------------------------

mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 8, 1 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels {
    count: 2
    policy: RELAXED
  }
}

# --- Instantiation ----------------------------------------------------------
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
"""

# Mesh graph descriptor template for Galaxy (non-lb) partitions
GALAXY_MESH_GRAPH_TEMPLATE = """# --- Meshes ---------------------------------------------------------------

mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 32, 1 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: RELAXED }
}

# --- Pinnings ---------------------------------------------------------------

pinnings {
  logical_fabric_node_id {
    mesh_id: 0
    chip_id: 0
  }
  physical_asic_position {
    tray_id: 1
    asic_location: 1
  }
}

# --- Instantiation ----------------------------------------------------------
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
"""

SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err
{nodelist_directive}

# Set environmental variables
export HOME="/data/${{USER}}"
export TT_METAL_HOME="/data/${{USER}}/tt-metal"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export PYTHONPATH="${{TT_METAL_HOME}}"
source ${{TT_METAL_HOME}}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
export TT_MESH_GRAPH_DESC_PATH="{mesh_graph_desc_path}"
export TT_TRAIN_OVERRIDES_PATH="{training_overrides_path}"

# Reset devices
{reset_command}

# Change to output directory for output files
cd {output_dir}

# Run the training script
python {script_path}
"""


class JobManager:
    """Manages SLURM job submission, monitoring, and tracking."""

    def __init__(self, jobs_base_dir: Optional[str] = None):
        """Initialize the job manager.

        Args:
            jobs_base_dir: Base directory for job output files.
                          Defaults to ./jobs relative to this file.
        """
        if jobs_base_dir is None:
            self.jobs_base_dir = Path(__file__).parent / "jobs"
        else:
            self.jobs_base_dir = Path(jobs_base_dir)

        self.jobs_base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_cache: Dict[str, JobInfo] = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        """Load existing job information from disk."""
        if not self.jobs_base_dir.exists():
            return

        for job_dir in self.jobs_base_dir.iterdir():
            if job_dir.is_dir() and job_dir.name.startswith("job_"):
                status_file = job_dir / "status.json"
                if status_file.exists():
                    try:
                        with open(status_file, "r") as f:
                            data = json.load(f)
                            job_info = JobInfo.from_dict(data)
                            self._jobs_cache[job_info.job_id] = job_info
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not load job from {status_file}: {e}")

    def _save_job_status(self, job_info: JobInfo) -> None:
        """Save job status to disk."""
        job_dir = Path(job_info.output_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        status_file = job_dir / "status.json"
        with open(status_file, "w") as f:
            json.dump(job_info.to_dict(), f, indent=2)

    def get_available_partitions(self) -> List[Dict]:
        """Get available SLURM partitions.

        Returns:
            List of partition info dictionaries with name, description, etc.
            Aggregates multiple sinfo lines per partition; has_free_nodes is True
            if any node group is idle or mix.
        """
        partitions = []
        partition_by_name: Dict[str, Dict] = {}

        try:
            result = subprocess.run(
                ["sinfo", "--noheader", "--format=%P|%a|%D|%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("|")
                    if len(parts) >= 4:
                        name = parts[0].rstrip("*")
                        avail = parts[1]
                        nodes = parts[2]
                        state = (parts[3] or "").lower()

                        if name not in partition_by_name:
                            partition_info = {
                                "name": name,
                                "available": avail == "up",
                                "nodes": nodes,
                                "state": state,
                                "has_free_nodes": "idle" in state or "mix" in state,
                            }
                            if name in PARTITION_DEVICE_MAPPING:
                                partition_info.update(PARTITION_DEVICE_MAPPING[name])
                            else:
                                partition_info["mesh_shape"] = [1, 1]
                                partition_info["description"] = f"{name} (unknown config)"
                                partition_info["max_nodes"] = 1
                            partition_by_name[name] = partition_info
                        else:
                            if "idle" in state or "mix" in state:
                                partition_by_name[name]["has_free_nodes"] = True

                partitions = list(partition_by_name.values())

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Warning: Could not query SLURM partitions: {e}")

        if not partitions:
            for name, info in PARTITION_DEVICE_MAPPING.items():
                partitions.append(
                    {
                        "name": name,
                        "available": True,
                        "nodes": "N/A",
                        "state": "unknown",
                        "has_free_nodes": False,
                        **info,
                    }
                )

        return partitions

    def generate_slurm_script(
        self,
        config: Dict,
        partition: str,
        nodes: int,
        job_name: str,
        output_dir: Path,
    ) -> str:
        """Generate a SLURM batch script.

        Args:
            config: Training configuration dictionary
            partition: SLURM partition name
            nodes: Number of nodes to request
            job_name: Name for the job
            output_dir: Directory for job outputs

        Returns:
            Generated SLURM script content
        """
        # Use script path relative to tt_metal_runtime_root for consistent resolution
        tt_train_root = f"{get_tt_metal_runtime_root()}/tt-train"
        script_path = Path(f"{tt_train_root}/sources/examples/gsm8k_finetune/gsm8k_finetune.py")

        # If the script doesn't exist at the expected location, try relative to output_dir
        # (for cases where jobs are run from /data and script is copied there)
        if not script_path.exists():
            script_path = output_dir.parent.parent / "gsm8k_finetune.py"
        if not script_path.exists():
            script_path = Path(__file__).parent / "gsm8k_finetune.py"

        is_lb_partition = "lb" in partition.lower()

        # Choose reset command based on partition type
        if is_lb_partition:
            reset_command = "tt-smi -r"
        else:
            reset_command = "tt-smi -glx_reset"

        # Nodelist: omit by default so SLURM can schedule on any free node (job queues).
        # Set TT_USE_NODELIST=1 to pin to GALAXY_NODES for non-lb partitions.
        nodelist_directive = ""
        if not is_lb_partition and os.environ.get("TT_USE_NODELIST") == "1":
            nodelist_directive = f"#SBATCH --nodelist={','.join(GALAXY_NODES)}"

        # Paths to job-specific config files
        mesh_graph_desc_path = output_dir / "mesh_graph_descriptor.textproto"
        training_overrides_path = output_dir / "training_overrides.yaml"

        script_content = SLURM_SCRIPT_TEMPLATE.format(
            partition=partition,
            nodes=nodes,
            job_name=job_name,
            output_dir=str(output_dir),
            reset_command=reset_command,
            script_path=str(script_path),
            nodelist_directive=nodelist_directive,
            mesh_graph_desc_path=str(mesh_graph_desc_path),
            training_overrides_path=str(training_overrides_path),
        )

        return script_content

    def generate_mesh_graph_descriptor(self, partition: str, output_dir: Path) -> Path:
        """Generate a mesh graph descriptor file for the job.

        Args:
            partition: SLURM partition name
            output_dir: Directory to write the descriptor file

        Returns:
            Path to the created mesh graph descriptor file
        """
        is_lb_partition = "lb" in partition.lower()

        if is_lb_partition:
            content = LB_MESH_GRAPH_TEMPLATE
        else:
            content = GALAXY_MESH_GRAPH_TEMPLATE

        descriptor_path = output_dir / "mesh_graph_descriptor.textproto"
        with open(descriptor_path, "w") as f:
            f.write(content)

        return descriptor_path

    def create_training_overrides(self, config: Dict, output_dir: Path) -> Path:
        """Create training_overrides.yaml for the job.

        Args:
            config: Training configuration dictionary
            output_dir: Directory to write the config file

        Returns:
            Path to the created config file
        """
        import yaml

        training_config = {
            "batch_size": config.get("batch_size", 32),
            "validation_batch_size": config.get("validation_batch_size", 4),
            "max_steps": config.get("max_steps", 60),
            "gradient_accumulation_steps": config.get("gradient_accumulation", 1),
            "eval_every": config.get("eval_every", 20),
        }

        # Add model_config if specified
        if config.get("model_config"):
            training_config["model_config"] = config["model_config"]
        # Add dataset (HF name or URL) for training script
        if config.get("dataset"):
            training_config["dataset"] = config["dataset"]

        scheduler_config = {
            "warmup_steps": config.get("warmup_steps", 20),
            "hold_steps": config.get("hold_steps", 40),
            "min_lr": config.get("min_lr", 3e-5),
            "max_lr": config.get("max_lr", 1e-4),
        }

        device_config = {
            "enable_ddp": config.get("enable_ddp", False),
            "mesh_shape": config.get("mesh_shape", [1, 1]),
        }

        transformer_config = {
            "max_sequence_length": config.get("max_seq_length", 512),
        }

        config_path = output_dir / "training_overrides.yaml"

        with open(config_path, "w") as f:
            f.write("training_config:\n")
            for key, value in training_config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\ntransformer_config:\n")
            for key, value in transformer_config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nscheduler_config:\n")
            for key, value in scheduler_config.items():
                if isinstance(value, float) and (value < 0.001 or value > 1000):
                    f.write(f"  {key}: {value:.2e}\n")
                else:
                    f.write(f"  {key}: {value}\n")

            f.write("\ndevice_config:\n")
            f.write(f"  enable_ddp: {str(device_config['enable_ddp']).lower()}\n")
            mesh_str = yaml.dump(device_config["mesh_shape"], default_flow_style=True).strip()
            f.write(f"  mesh_shape: {mesh_str}\n")

        return config_path

    def submit_job(
        self,
        config: Dict,
        partition: str,
        nodes: int = 1,
        job_name: Optional[str] = None,
    ) -> tuple[bool, str, Optional[JobInfo]]:
        """Submit a training job to SLURM.

        Args:
            config: Training configuration dictionary
            partition: SLURM partition name
            nodes: Number of nodes to request
            job_name: Optional job name (auto-generated if not provided)

        Returns:
            Tuple of (success, message, job_info)
        """
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"gsm8k_{timestamp}"

        job_dir = self.jobs_base_dir / f"job_{job_name}"
        job_dir.mkdir(parents=True, exist_ok=True)

        # Determine mesh shape and node selection based on partition
        is_lb_partition = "lb" in partition.lower()
        if is_lb_partition:
            mesh_shape = [8, 1]
        else:
            mesh_shape = [32, 1]

        # Update config with correct mesh shape
        config = config.copy()
        config["mesh_shape"] = mesh_shape
        config["enable_ddp"] = True

        try:
            # Create job-specific training overrides
            self.create_training_overrides(config, job_dir)

            # Generate mesh graph descriptor
            self.generate_mesh_graph_descriptor(partition, job_dir)

        except Exception as e:
            return False, f"Failed to create config: {e}", None

        script_content = self.generate_slurm_script(config, partition, nodes, job_name, job_dir)
        script_path = job_dir / "submit.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        with open(job_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return (
                    False,
                    f"sbatch failed: {result.stderr or result.stdout}",
                    None,
                )

            job_id = result.stdout.strip().split()[-1]

            job_info = JobInfo(
                job_id=job_id,
                job_name=job_name,
                partition=partition,
                nodes=nodes,
                status=JobStatus.PENDING.value,
                submit_time=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                config=config,
                output_dir=str(job_dir),
            )

            self._jobs_cache[job_id] = job_info
            self._save_job_status(job_info)

            return True, f"Job submitted successfully! Job ID: {job_id}", job_info

        except subprocess.TimeoutExpired:
            return False, "sbatch timed out", None
        except FileNotFoundError:
            return False, "sbatch command not found. Is SLURM installed?", None
        except Exception as e:
            return False, f"Error submitting job: {e}", None

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the current status of a job.

        Args:
            job_id: SLURM job ID

        Returns:
            JobStatus enum value
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "--noheader", "--format=%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                status_str = result.stdout.strip().upper()
                if status_str in ["PENDING", "PD"]:
                    return JobStatus.PENDING
                elif status_str in ["RUNNING", "R"]:
                    return JobStatus.RUNNING
                elif status_str in ["COMPLETED", "CD"]:
                    return JobStatus.COMPLETED
                elif status_str in ["FAILED", "F"]:
                    return JobStatus.FAILED
                elif status_str in ["CANCELLED", "CA"]:
                    return JobStatus.CANCELLED
                else:
                    return JobStatus.UNKNOWN
            else:
                # Job not in squeue - check sacct. Use -S to look back 30 days
                # (default sacct window may exclude older completed/cancelled jobs).
                # Use -X to get job allocation only (not individual steps).
                start_str = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
                result = subprocess.run(
                    [
                        "sacct",
                        "-j",
                        job_id,
                        "-S",
                        start_str,
                        "-X",  # job allocation only
                        "--noheader",
                        "--format=State",
                        "--parsable2",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0 and result.stdout.strip():
                    status_str = result.stdout.strip().split("\n")[0].upper()
                    if "COMPLETED" in status_str or status_str == "CD":
                        return JobStatus.COMPLETED
                    elif (
                        "FAILED" in status_str
                        or "TIMEOUT" in status_str
                        or "OUT_OF_MEMORY" in status_str
                        or "NODE_FAIL" in status_str
                        or status_str == "F"
                    ):
                        return JobStatus.FAILED
                    elif "CANCELLED" in status_str or status_str == "CA":
                        return JobStatus.CANCELLED
                    elif "RUNNING" in status_str or status_str == "R":
                        return JobStatus.RUNNING
                    elif "PENDING" in status_str or status_str == "PD":
                        return JobStatus.PENDING

                return JobStatus.UNKNOWN

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return JobStatus.UNKNOWN

    def cancel_job(self, job_id: str) -> tuple[bool, str]:
        """Cancel a running or pending job.

        Args:
            job_id: SLURM job ID

        Returns:
            Tuple of (success, message)
        """
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                if job_id in self._jobs_cache:
                    self._jobs_cache[job_id].status = JobStatus.CANCELLED.value
                    self._jobs_cache[job_id].end_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                    self._save_job_status(self._jobs_cache[job_id])
                return True, f"Job {job_id} cancelled successfully"
            else:
                return False, f"Failed to cancel job: {result.stderr or result.stdout}"

        except subprocess.TimeoutExpired:
            return False, "scancel timed out"
        except FileNotFoundError:
            return False, "scancel command not found"
        except Exception as e:
            return False, f"Error cancelling job: {e}"

    def update_job_statuses(self) -> None:
        """Update status of all tracked jobs."""
        for job_id, job_info in self._jobs_cache.items():
            if job_info.status in [
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ]:
                continue

            new_status = self.get_job_status(job_id)
            if new_status.value != job_info.status:
                job_info.status = new_status.value

                if new_status == JobStatus.RUNNING and job_info.start_time is None:
                    job_info.start_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

                if new_status in [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ]:
                    job_info.end_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

                self._save_job_status(job_info)

    def get_all_jobs(self) -> List[JobInfo]:
        """Get all tracked jobs.

        Returns:
            List of JobInfo objects
        """
        self.update_job_statuses()
        return list(self._jobs_cache.values())

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get a specific job by ID.

        Args:
            job_id: SLURM job ID

        Returns:
            JobInfo if found, None otherwise
        """
        if job_id in self._jobs_cache:
            self.update_job_statuses()
            return self._jobs_cache.get(job_id)
        return None

    def get_job_output_file(self, job_id: str, filename: str) -> Optional[Path]:
        """Get path to a job's output file.

        Args:
            job_id: SLURM job ID
            filename: Name of the output file (e.g., 'output.txt')

        Returns:
            Path to the file if it exists, None otherwise
        """
        job_info = self._jobs_cache.get(job_id)
        if job_info:
            file_path = Path(job_info.output_dir) / filename
            if file_path.exists():
                return file_path
        return None

    def delete_job(self, job_id: str, cancel_if_running: bool = True) -> tuple[bool, str]:
        """Delete a job and its output directory.

        Args:
            job_id: SLURM job ID
            cancel_if_running: If True, cancel the job if it's still running

        Returns:
            Tuple of (success, message)
        """
        job_info = self._jobs_cache.get(job_id)
        if not job_info:
            return False, f"Job {job_id} not found"

        if job_info.status in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
            if cancel_if_running:
                success, msg = self.cancel_job(job_id)
                if not success:
                    return False, f"Failed to cancel job before deletion: {msg}"
            else:
                return (
                    False,
                    "Job is still running. Set cancel_if_running=True to cancel first.",
                )

        try:
            job_dir = Path(job_info.output_dir)
            if job_dir.exists():
                shutil.rmtree(job_dir)

            del self._jobs_cache[job_id]
            return True, f"Job {job_id} deleted successfully"

        except Exception as e:
            return False, f"Error deleting job: {e}"
