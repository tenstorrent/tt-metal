# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
from abc import ABC, abstractmethod

from framework.sweeps_logger import sweeps_logger as logger


class VectorSource(ABC):
    """Abstract base class for test vector sources"""

    @abstractmethod
    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors based on criteria"""
        pass

    @abstractmethod
    def get_available_suites(self, module_name: str) -> list[str]:
        """Get list of available suites for a module"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the source is accessible"""
        pass


class FileVectorSource(VectorSource):
    """File-based vector source"""

    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors from JSON file"""
        if not self.file_path.exists():
            return []

        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)

            vectors = []

            if vector_id:
                # Find specific vector by ID
                for suite_key, suite_content in data.items():
                    if vector_id in suite_content:
                        vector = suite_content[vector_id]
                        vector["input_hash"] = vector_id
                        vector["suite_name"] = suite_key
                        # Preserve stored sweep_name (may include mesh suffix), fallback to module_name
                        if "sweep_name" not in vector:
                            vector["sweep_name"] = module_name
                        vectors.append(vector)
                        break
            else:
                # Load by suite or all suites
                for suite_key, suite_content in data.items():
                    if suite_name and suite_name != suite_key:
                        continue

                    for input_hash, vector_data in suite_content.items():
                        vector_data["input_hash"] = input_hash
                        vector_data["suite_name"] = suite_key
                        # Preserve stored sweep_name (may include mesh suffix), fallback to module_name
                        if "sweep_name" not in vector_data:
                            vector_data["sweep_name"] = module_name
                        vectors.append(vector_data)

            return vectors

        except (json.JSONDecodeError, IOError):
            return []

    def get_available_suites(self, module_name: str) -> list[str]:
        """Get list of available suites from JSON file"""
        if not self.file_path.exists():
            return []

        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)
            return list(data.keys())
        except (json.JSONDecodeError, IOError):
            return []

    def validate_connection(self) -> bool:
        """Validate that the file exists and is readable"""
        return self.file_path.exists() and self.file_path.is_file()


class VectorExportSource(VectorSource):
    """Vectors export directory source"""

    def __init__(self, export_dir: pathlib.Path | None = None):
        if export_dir is None:
            # Default to vectors_export directory relative to this file
            self.export_dir = pathlib.Path(__file__).parent.parent / "vectors_export"
        else:
            self.export_dir = export_dir

    def _find_module_files(self, module_name: str) -> list[pathlib.Path]:
        """Find all JSON files for a given module (including mesh variants)"""
        # First try exact match (backward compatibility)
        exact_match = list(self.export_dir.glob(f"{module_name}.json"))
        if exact_match:
            return exact_match

        # Then look for mesh-suffixed variants (e.g., module__mesh_2x4.json)
        mesh_variants = list(self.export_dir.glob(f"{module_name}__mesh_*.json"))
        if mesh_variants:
            logger.info(f"Found {len(mesh_variants)} mesh variant file(s) for module '{module_name}'")
            return sorted(mesh_variants)  # Sort for consistent ordering

        logger.warning(f"No vector file found for module '{module_name}' in {self.export_dir}")
        try:
            tail = module_name.split(".")[-1]
            similar_files = list(self.export_dir.glob(f"*{tail}*.json"))
            if similar_files:
                top_names = [f.name for f in similar_files[:5]]
                logger.info(f"Similar files found: {top_names}")
        except Exception:
            pass
        return []

    def _get_machine_info(self):
        """Get machine info using get_machine_info from generic_ops_tracer."""
        try:
            import sys
            import os

            # Add model_tracer to path if not already there
            # Go up 4 levels from this file (tests/sweep_framework/framework/vector_source.py)
            # to get to repo root, then into model_tracer
            model_tracer_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "model_tracer"
            model_tracer_path_str = str(model_tracer_path)

            if model_tracer_path_str not in sys.path:
                sys.path.insert(0, model_tracer_path_str)

            # Import the module
            from generic_ops_tracer import get_machine_info

            machine_info = get_machine_info()

            # get_machine_info() might return None if tt-smi fails
            if machine_info is None:
                logger.warning("get_machine_info() returned None - tt-smi might have failed")
                return None

            logger.debug(f"Successfully retrieved machine info: {machine_info}")
            return machine_info
        except Exception as e:
            import traceback

            logger.warning(f"Failed to get machine info: {e}\n{traceback.format_exc()}")
            return None

    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors from vectors_export directory (including mesh variants)

        If MESH_DEVICE_SHAPE environment variable is set, filters vectors to only load
        those matching the current machine's configuration.
        """
        import os

        module_files = self._find_module_files(module_name)
        if not module_files:
            return []

        # Check if this is a model_traced run (resource filtering only applies to model_traced)
        is_model_traced = "model_traced" in module_name

        # Get current machine info (for device/card filtering in model_traced runs)
        current_machine_info = None
        if is_model_traced:
            current_machine_info = self._get_machine_info()

        # Check if mesh filtering is enabled via environment variable
        mesh_filter = os.environ.get("MESH_DEVICE_SHAPE", "").strip()
        target_mesh = None

        if mesh_filter:
            logger.info(f"Mesh filtering enabled: MESH_DEVICE_SHAPE={mesh_filter}")

            if current_machine_info:
                logger.info(
                    f"Current machine: board_type={current_machine_info['board_type']}, "
                    f"device_series={current_machine_info['device_series']}, "
                    f"card_count={current_machine_info['card_count']}"
                )
            else:
                logger.warning("Could not determine current machine info from tt-smi")

            # Parse target mesh shape from env var (e.g., "1x2" -> (1, 2))
            try:
                target_rows, target_cols = map(int, mesh_filter.lower().split("x"))
                target_mesh = (target_rows, target_cols)
            except (ValueError, AttributeError):
                logger.warning(f"Invalid MESH_DEVICE_SHAPE format: {mesh_filter}, expected NxM (e.g., 1x2)")

        all_vectors = []
        filtered_count = 0
        machine_mismatch_count = 0

        # Load vectors from all matching files (e.g., base + mesh variants)
        for module_file in module_files:
            try:
                with open(module_file, "r") as file:
                    data = json.load(file)

                for suite_key, suite_content in data.items():
                    if suite_name and suite_name != suite_key:
                        continue

                    if vector_id:
                        if vector_id in suite_content:
                            vector = suite_content[vector_id]
                            vector["input_hash"] = vector_id
                            vector["suite_name"] = suite_key
                            # Preserve stored sweep_name (may include mesh suffix), fallback to module_name
                            if "sweep_name" not in vector:
                                vector["sweep_name"] = module_name
                            all_vectors.append(vector)
                            logger.info(
                                f"Vector ID '{vector_id}' found in suite '{suite_name}' of module '{module_name}' (file: {module_file.name})"
                            )
                        else:
                            logger.warning(
                                f"Vector ID '{vector_id}' not found in suite '{suite_name}' of module '{module_name}' (file: {module_file.name})"
                            )
                        break
                    else:
                        for input_hash, vector_data in suite_content.items():
                            vector_data["input_hash"] = input_hash
                            vector_data["suite_name"] = suite_key
                            # Preserve stored sweep_name (may include mesh suffix), fallback to module_name
                            if "sweep_name" not in vector_data:
                                vector_data["sweep_name"] = module_name

                            # Get traced_machine_info for filtering checks
                            traced_machine_info = vector_data.get("traced_machine_info")
                            # Handle both list and dict formats
                            if isinstance(traced_machine_info, list) and traced_machine_info:
                                traced_machine_info = traced_machine_info[0]

                            # Check if required mesh shape exceeds available devices
                            # This check only applies to model_traced runs (not nightly/lead models)
                            # and is independent of MESH_DEVICE_SHAPE env var
                            skip_for_resources = False
                            if current_machine_info and traced_machine_info and isinstance(traced_machine_info, dict):
                                # Get the mesh shape from traced config (this is what actually matters)
                                traced_mesh_shape = traced_machine_info.get("mesh_device_shape")

                                # Calculate required device count from mesh shape
                                if isinstance(traced_mesh_shape, list) and len(traced_mesh_shape) == 2:
                                    required_device_count = traced_mesh_shape[0] * traced_mesh_shape[1]
                                else:
                                    # Fallback to device_count if mesh_shape not available
                                    required_device_count = traced_machine_info.get("device_count", 1)

                                # Get current machine capabilities
                                current_device_count = current_machine_info.get("device_count", 1)

                                # Skip if vector requires more devices than available
                                # Use mesh shape product (actual requirement) instead of card_count from trace machine
                                if required_device_count > current_device_count:
                                    logger.debug(
                                        f"Skipping vector requiring {required_device_count} devices "
                                        f"(mesh shape: {traced_mesh_shape}) "
                                        f"(current machine has {current_device_count} devices)"
                                    )
                                    machine_mismatch_count += 1
                                    skip_for_resources = True

                            if skip_for_resources:
                                continue

                            # Apply mesh filtering if enabled
                            if mesh_filter and target_mesh:
                                if traced_machine_info and isinstance(traced_machine_info, dict):
                                    # Extract mesh shape from traced config
                                    vector_mesh = traced_machine_info.get("mesh_device_shape")
                                    if isinstance(vector_mesh, list) and len(vector_mesh) == 2:
                                        vector_mesh_tuple = (vector_mesh[0], vector_mesh[1])
                                    else:
                                        vector_mesh_tuple = (1, 1)  # Default for single device

                                    # Check if mesh shape matches
                                    if vector_mesh_tuple != target_mesh:
                                        filtered_count += 1
                                        continue

                                    # Validate device_count consistency
                                    device_count = traced_machine_info.get("device_count", 1)
                                    expected_device_count = target_mesh[0] * target_mesh[1]
                                    if device_count != expected_device_count:
                                        logger.debug(
                                            f"Vector mesh {vector_mesh_tuple} has device_count={device_count}, "
                                            f"expected {expected_device_count} for mesh {target_mesh}"
                                        )
                                        filtered_count += 1
                                        continue

                                    # Check machine compatibility if current_machine_info is available
                                    if current_machine_info:
                                        # Check board_type (flexible matching for wormhole variants)
                                        traced_board = traced_machine_info.get("board_type", "").lower()
                                        current_board = current_machine_info.get("board_type", "").lower()
                                        if traced_board and current_board:
                                            # Allow "wormhole" to match "wormhole_b0" etc.
                                            board_match = (
                                                traced_board == current_board
                                                or "wormhole" in traced_board
                                                and "wormhole" in current_board
                                            )
                                            if not board_match:
                                                machine_mismatch_count += 1
                                                continue

                                        # Check device_series
                                        traced_series = traced_machine_info.get("device_series", "").lower()
                                        current_series = current_machine_info.get("device_series", "").lower()
                                        if traced_series and current_series and traced_series != current_series:
                                            machine_mismatch_count += 1
                                            continue

                            all_vectors.append(vector_data)

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading vectors from {module_file}: {e}")

        # Log filtering results if filtering was enabled
        if mesh_filter and (filtered_count > 0 or machine_mismatch_count > 0):
            total_filtered = filtered_count + machine_mismatch_count
            logger.info(
                f"Filtered out {total_filtered} vectors "
                f"(mesh mismatch: {filtered_count}, machine mismatch: {machine_mismatch_count}), "
                f"loaded {len(all_vectors)} vectors"
            )

        return all_vectors

    def get_available_suites(self, module_name: str) -> list[str]:
        """Get list of available suites for a module from vectors_export directory (including mesh variants)"""
        module_files = self._find_module_files(module_name)
        if not module_files:
            return []

        # Collect unique suite names across all mesh variant files
        all_suites = set()
        for module_file in module_files:
            try:
                with open(module_file, "r") as file:
                    data = json.load(file)
                all_suites.update(data.keys())
            except (json.JSONDecodeError, IOError):
                continue

        return sorted(list(all_suites))

    def validate_connection(self) -> bool:
        """Validate that the export directory exists"""
        return self.export_dir.exists() and self.export_dir.is_dir()


class VectorSourceFactory:
    """Factory to create appropriate vector source based on configuration"""

    SUPPORTED_SOURCES = {"file", "vectors_export"}

    @staticmethod
    def create_source(vector_source: str, **kwargs) -> VectorSource:
        if vector_source == "file":
            if "file_path" not in kwargs:
                raise ValueError("Missing required argument 'file_path' for file vector source")
            return FileVectorSource(kwargs["file_path"])
        elif vector_source == "vectors_export":
            export_dir = kwargs.get("export_dir")
            return VectorExportSource(export_dir)
        else:
            raise ValueError(
                f"Unknown vector source: '{vector_source}'. "
                f"Supported sources: {', '.join(sorted(VectorSourceFactory.SUPPORTED_SOURCES))}"
            )
