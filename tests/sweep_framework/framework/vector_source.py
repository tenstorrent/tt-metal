# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
import pathlib
from abc import ABC, abstractmethod

from .constants import parse_hardware_suffix, parse_mesh_suffix, strip_grouping_suffix
from .matrix_runner_config import (
    GENERATION_MANIFEST_FILENAME,
    SUPPORTED_VECTOR_GROUPING_MODES,
    get_allowed_mesh_shapes_for_local_hardware_group,
    get_lead_models_test_group_name_for_hardware_group,
    get_test_group_capability_profile,
    get_test_group_name_for_hardware_group,
    get_vector_load_filter_policy,
    hardware_group_matches_any_rule,
)
from .sweeps_logger import sweeps_logger as logger


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
        self._cached_generation_manifest = None
        self._cached_manifest_vector_paths = None

    def _load_generation_manifest(self) -> dict:
        """Load and validate the generation manifest used as the sole file index."""
        if self._cached_generation_manifest is not None:
            return self._cached_generation_manifest

        manifest_path = self.export_dir / GENERATION_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing generation manifest at {manifest_path}. "
                "vectors_export lookups require generation_manifest.json."
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to read generation manifest at {manifest_path}: {e}") from e

        if not isinstance(manifest, dict):
            raise RuntimeError(f"Generation manifest at {manifest_path} is not a JSON object")

        grouping_mode = manifest.get("vector_grouping_mode")
        if grouping_mode not in SUPPORTED_VECTOR_GROUPING_MODES:
            raise RuntimeError(
                f"Generation manifest at {manifest_path} must contain vector_grouping_mode set to 'mesh' or 'hw'"
            )

        vector_files = manifest.get("vector_files")
        if not isinstance(vector_files, list) or not vector_files:
            raise RuntimeError(f"Generation manifest at {manifest_path} must contain a non-empty 'vector_files' list")

        validated_vector_files = []
        for index, file_name in enumerate(vector_files):
            if not isinstance(file_name, str) or not file_name.endswith(".json"):
                raise RuntimeError(
                    f"Generation manifest at {manifest_path} has invalid vector_files[{index}]={file_name!r}"
                )
            if pathlib.Path(file_name).name == GENERATION_MANIFEST_FILENAME:
                raise RuntimeError(
                    f"Generation manifest at {manifest_path} must not list {GENERATION_MANIFEST_FILENAME} as a vector file"
                )
            validated_vector_files.append(file_name)

        manifest["vector_files"] = validated_vector_files
        self._cached_generation_manifest = manifest
        return manifest

    def _get_manifest_vector_paths(self) -> list[pathlib.Path]:
        """Resolve manifest vector file names to concrete existing paths."""
        if self._cached_manifest_vector_paths is not None:
            return self._cached_manifest_vector_paths

        manifest = self._load_generation_manifest()
        manifest_vector_paths = []

        for file_name in manifest["vector_files"]:
            vector_path = self.export_dir / file_name
            if not vector_path.exists():
                raise FileNotFoundError(
                    f"Generation manifest references missing vector file '{file_name}' in {self.export_dir}"
                )
            if not vector_path.is_file():
                raise RuntimeError(
                    f"Generation manifest entry '{file_name}' in {self.export_dir} does not resolve to a file"
                )
            manifest_vector_paths.append(vector_path)

        self._cached_manifest_vector_paths = manifest_vector_paths
        return manifest_vector_paths

    @staticmethod
    def _get_grouping_kind(module_name: str) -> str | None:
        """Classify a manifest module name by shared suffix parsing rules."""
        if parse_mesh_suffix(module_name) is not None:
            return "mesh"
        if parse_hardware_suffix(module_name) is not None:
            return "hw"
        return None

    def _manifest_entry_matches_module(
        self, requested_module_name: str, manifest_module_name: str, grouping_mode: str
    ) -> bool:
        """Apply one file-list rule for exact or grouped manifest entries."""
        if manifest_module_name == requested_module_name:
            return True

        if strip_grouping_suffix(requested_module_name) != requested_module_name:
            return False

        if strip_grouping_suffix(manifest_module_name) != requested_module_name:
            return False

        return self._get_grouping_kind(manifest_module_name) == grouping_mode

    def _find_module_files(self, module_name: str) -> list[pathlib.Path]:
        """Find all manifest-declared JSON files for a given module."""
        manifest = self._load_generation_manifest()
        manifest_vector_paths = self._get_manifest_vector_paths()
        grouping_mode = manifest["vector_grouping_mode"]
        module_files = sorted(
            {
                vector_path
                for vector_path in manifest_vector_paths
                if self._manifest_entry_matches_module(module_name, vector_path.stem, grouping_mode)
            }
        )

        if module_files:
            grouped_count = sum(1 for vector_path in module_files if vector_path.stem != module_name)
            if grouped_count:
                logger.info(
                    f"Resolved {grouped_count} grouped vector file(s) for module '{module_name}' from "
                    f"{GENERATION_MANIFEST_FILENAME} using grouping mode '{grouping_mode}'"
                )
            return module_files

        available_modules = sorted(vector_path.stem for vector_path in manifest_vector_paths)
        preview = available_modules[:5]
        raise FileNotFoundError(
            f"No vector file for module '{module_name}' found in generation manifest {self.export_dir / GENERATION_MANIFEST_FILENAME}. "
            f"Manifest grouping mode is '{grouping_mode}'. Available manifest entries include: {preview}"
        )

    @staticmethod
    def _parse_mesh_shape_string(mesh_shape: str) -> tuple[int, int] | None:
        """Parse 'rowsxcols' mesh shape strings."""
        try:
            rows, cols = map(int, mesh_shape.lower().split("x"))
            return (rows, cols)
        except (AttributeError, ValueError):
            return None

    def _get_run_type(self, module_name: str, is_lead_models: bool) -> str | None:
        """Infer sweep run type relevant for vector filtering."""
        if is_lead_models:
            return "lead_models"
        if "model_traced" in module_name:
            return "model_traced"
        return None

    def _get_current_hardware_group(self, current_machine_info: dict | None) -> tuple[str, str, int] | None:
        """Convert current machine info into the normalized hardware-group tuple."""
        if not current_machine_info:
            return None

        board_type = current_machine_info.get("board_type")
        device_series = current_machine_info.get("device_series")
        card_count = current_machine_info.get("card_count")
        if not board_type or not device_series or not isinstance(card_count, int):
            return None

        return (str(board_type).lower(), str(device_series).lower(), card_count)

    @staticmethod
    def _normalize_traced_machine_entries(vector_data: dict) -> list[dict]:
        """Normalize traced_machine_info to a list of machine entry dictionaries."""
        traced_machine_info = vector_data.get("traced_machine_info")
        if isinstance(traced_machine_info, dict):
            return [traced_machine_info]
        if isinstance(traced_machine_info, list):
            return [entry for entry in traced_machine_info if isinstance(entry, dict)]
        return []

    @staticmethod
    def _extract_mesh_shape(entry: dict) -> tuple[int, int] | None:
        """Extract mesh shape from machine entry when present."""
        mesh = entry.get("mesh_device_shape")
        if isinstance(mesh, list) and len(mesh) == 2:
            return (mesh[0], mesh[1])
        if isinstance(mesh, str):
            try:
                parsed = ast.literal_eval(mesh)
                if isinstance(parsed, list) and len(parsed) == 2:
                    return (parsed[0], parsed[1])
            except (SyntaxError, ValueError):
                # Treat unparseable mesh strings as missing mesh information.
                logger.debug("Failed to parse mesh_device_shape from traced_machine_info entry: %r", mesh)

        placements = entry.get("tensor_placements")
        if isinstance(placements, list) and placements:
            placement_mesh = placements[0].get("mesh_device_shape")
            if isinstance(placement_mesh, list) and len(placement_mesh) == 2:
                return (placement_mesh[0], placement_mesh[1])
            if isinstance(placement_mesh, str):
                try:
                    parsed = ast.literal_eval(placement_mesh)
                    if isinstance(parsed, list) and len(parsed) == 2:
                        return (parsed[0], parsed[1])
                except (SyntaxError, ValueError):
                    # Treat unparseable placement mesh strings as missing mesh information.
                    logger.debug(
                        "Failed to parse mesh_device_shape from tensor_placements entry: %r",
                        placement_mesh,
                    )
        return None

    @staticmethod
    def _get_entry_hardware_group(entry: dict) -> tuple[str | None, str | None, int | None] | None:
        """Convert traced machine entry to normalized hardware tuple with optional fields."""
        board_type = entry.get("board_type")
        device_series = entry.get("device_series")
        card_count = entry.get("card_count")

        if not board_type and not device_series and card_count is None:
            return None

        normalized_board = str(board_type).lower() if board_type else None
        normalized_series = str(device_series).lower() if device_series else None
        normalized_cards = card_count if isinstance(card_count, int) else None
        return (normalized_board, normalized_series, normalized_cards)

    def _resolve_runtime_test_group_name(self, run_type: str | None, current_machine_info: dict | None) -> str | None:
        """Determine the logical runner lane for CI or local execution.

        When TEST_GROUP_NAME is present, we are in a CI-like mode where the
        matrix has already chosen the owning lane. Without TEST_GROUP_NAME we
        fall back to hardware inference so local/manual runs still work.
        """
        explicit_test_group_name = os.environ.get("TEST_GROUP_NAME", "").strip()
        if explicit_test_group_name:
            return explicit_test_group_name

        hardware_group = self._get_current_hardware_group(current_machine_info)
        if hardware_group is None:
            return None

        if run_type == "lead_models":
            return get_lead_models_test_group_name_for_hardware_group(hardware_group)
        if run_type == "model_traced":
            return get_test_group_name_for_hardware_group(hardware_group)
        return None

    def _resolve_allowed_mesh_shapes(
        self, capability_profile: dict, current_machine_info: dict | None
    ) -> set[tuple[int, int]]:
        """Resolve allowed mesh shapes from explicit env, CI ownership, or local capability.

        Resolution order:
        1. ``MESH_DEVICE_SHAPE`` explicit override for manual targeting
        2. Strict CI ownership when ``TEST_GROUP_NAME`` is set
        3. Broader local hardware capability inferred from the current machine
        """
        mesh_filter = os.environ.get("MESH_DEVICE_SHAPE", "").strip()
        if mesh_filter:
            parsed = self._parse_mesh_shape_string(mesh_filter)
            if parsed is None:
                logger.warning(f"Invalid MESH_DEVICE_SHAPE format: {mesh_filter}, expected NxM (e.g., 1x2)")
                return set()
            return {parsed}

        explicit_test_group_name = os.environ.get("TEST_GROUP_NAME", "").strip()
        allowed_meshes = set()
        if explicit_test_group_name:
            mesh_shape_strings = capability_profile.get("allowed_mesh_shapes", ())
        else:
            current_hardware_group = self._get_current_hardware_group(current_machine_info)
            mesh_shape_strings = get_allowed_mesh_shapes_for_local_hardware_group(current_hardware_group)

        for mesh_shape in mesh_shape_strings:
            parsed = self._parse_mesh_shape_string(mesh_shape)
            if parsed is not None:
                allowed_meshes.add(parsed)
        return allowed_meshes

    def _get_machine_info(self):
        """Get machine info using get_machine_info from generic_ops_tracer."""
        try:
            import sys

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

            # Validate that board_type is a recognized arch name (e.g. "Wormhole",
            # "Blackhole"). Reject PCI addresses and device paths — an invalid
            # board type would cause lead-model strict matching to filter out
            # every vector.
            board = machine_info.get("board_type", "")
            if not board or ":" in board or "/" in board:
                logger.warning(
                    f"get_machine_info() returned suspicious board_type='{board}' "
                    f"— ignoring machine info to avoid incorrect hardware filtering."
                )
                return None

            # Validate that card_count is present and usable. Downstream filtering
            # assumes this is known; if it is missing or invalid, disable
            # machine-based filtering by returning None.
            card_count = machine_info.get("card_count")
            if not isinstance(card_count, int) or card_count <= 0:
                logger.warning(
                    f"get_machine_info() returned invalid card_count='{card_count}' "
                    f"— ignoring machine info to avoid incorrect hardware filtering."
                )
                return None

            # Optionally sanity-check device_count if provided: if present but
            # invalid, treat machine info as unusable.
            if "device_count" in machine_info:
                device_count = machine_info.get("device_count")
                if not isinstance(device_count, int) or device_count <= 0:
                    logger.warning(
                        f"get_machine_info() returned invalid device_count='{device_count}' "
                        f"— ignoring machine info to avoid incorrect hardware filtering."
                    )
                    return None
            logger.debug(f"Successfully retrieved machine info: {machine_info}")
            return machine_info
        except Exception as e:
            import traceback

            logger.warning(f"Failed to get machine info: {e}\n{traceback.format_exc()}")
            return None

    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors from vectors_export directory (including grouped variants)

        If MESH_DEVICE_SHAPE environment variable is set, filters vectors to only load
        those matching the current machine's configuration.
        """
        import os

        manifest = self._load_generation_manifest()
        module_files = self._find_module_files(module_name)
        if not module_files:
            return []

        # Determine which traced sweep mode this module belongs to.
        is_lead_models = os.environ.get("LEAD_MODELS_RUN", "").strip() == "1"
        run_type = self._get_run_type(module_name, is_lead_models)
        filter_policy = get_vector_load_filter_policy(manifest["vector_grouping_mode"])
        filter_kind = filter_policy["kind"]

        # Fetch runtime machine info for traced sweep modes based on the resolved
        # run type rather than a module-name substring. This keeps the behavior
        # stable even if traced module naming changes later.
        current_machine_info = None
        if run_type in {"model_traced", "lead_models"}:
            current_machine_info = self._get_machine_info()

        test_group_name = self._resolve_runtime_test_group_name(run_type, current_machine_info)
        explicit_test_group_name = os.environ.get("TEST_GROUP_NAME", "").strip()
        capability_profile = (
            get_test_group_capability_profile(run_type, test_group_name)
            if test_group_name
            else {"allowed_mesh_shapes": (), "hardware_rules": ()}
        )

        if filter_kind in {"hardware", "mesh"} and current_machine_info:
            logger.info(
                f"Current machine: board_type={current_machine_info['board_type']}, "
                f"device_series={current_machine_info['device_series']}, "
                f"card_count={current_machine_info['card_count']}"
            )

        allowed_mesh_shapes = set()
        if filter_policy["enforce_mesh_capability"]:
            allowed_mesh_shapes = self._resolve_allowed_mesh_shapes(capability_profile, current_machine_info)
            if allowed_mesh_shapes:
                mesh_labels = sorted(f"{rows}x{cols}" for rows, cols in allowed_mesh_shapes)
                mode_label = "CI ownership" if explicit_test_group_name else "local hardware capability"
                logger.info(
                    f"Manifest-selected mesh filtering enabled for module '{module_name}' via {mode_label}: "
                    f"{mesh_labels}"
                )
            else:
                logger.warning(
                    f"Manifest grouping mode is 'mesh' for module '{module_name}', but no mesh capability was "
                    "derived. Set TEST_GROUP_NAME for strict CI ownership or MESH_DEVICE_SHAPE for a manual override."
                )

        strict_ci_mesh_ownership = filter_policy["enforce_mesh_capability"] and bool(explicit_test_group_name)

        hardware_rules = capability_profile.get("hardware_rules", ())
        if filter_policy["enforce_hardware_capability"] and not hardware_rules:
            logger.warning(
                f"Manifest grouping mode is 'hw' for module '{module_name}', but no hardware capability profile "
                "was derived. Set TEST_GROUP_NAME or run on a system whose hardware can infer one."
            )

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

                            traced_machine_entries = self._normalize_traced_machine_entries(vector_data)

                            if (
                                filter_policy["enforce_hardware_capability"]
                                and hardware_rules
                                and traced_machine_entries
                            ):
                                has_matching_hardware = any(
                                    hardware_group_matches_any_rule(
                                        self._get_entry_hardware_group(entry),
                                        hardware_rules,
                                    )
                                    for entry in traced_machine_entries
                                )

                                if not has_matching_hardware:
                                    logger.debug(
                                        f"Skipping vector - traced hardware is outside the capability profile for "
                                        f"test_group_name='{test_group_name}'"
                                    )
                                    machine_mismatch_count += 1
                                    continue

                            # Apply mesh filtering when manifest grouping mode says ownership is by mesh.
                            if filter_policy["enforce_mesh_capability"] and (
                                allowed_mesh_shapes or strict_ci_mesh_ownership
                            ):
                                # If mesh shape is missing in JSON, do not filter out.
                                # Otherwise, accept when ANY traced entry matches an allowed mesh.
                                traced_mesh_entries = []
                                for entry in traced_machine_entries:
                                    mesh_shape = self._extract_mesh_shape(entry)
                                    if mesh_shape is not None:
                                        traced_mesh_entries.append((entry, mesh_shape))

                                if traced_mesh_entries:
                                    matching_entries = [
                                        entry
                                        for entry, mesh_shape in traced_mesh_entries
                                        if mesh_shape in allowed_mesh_shapes
                                    ]
                                    if not matching_entries:
                                        filtered_count += 1
                                        continue

                            all_vectors.append(vector_data)

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading vectors from {module_file}: {e}")

        # Log filtering results when manifest-selected compatibility filtering applied.
        if filter_kind in {"hardware", "mesh"} and (filtered_count > 0 or machine_mismatch_count > 0):
            total_filtered = filtered_count + machine_mismatch_count
            logger.info(
                f"Filtered out {total_filtered} vectors "
                f"(mesh mismatch: {filtered_count}, machine mismatch: {machine_mismatch_count}), "
                f"loaded {len(all_vectors)} vectors"
            )

        return all_vectors

    def get_available_suites(self, module_name: str) -> list[str]:
        """Get list of available suites for a module from vectors_export directory (including grouped variants)."""
        module_files = self._find_module_files(module_name)
        if not module_files:
            return []

        # Collect unique suite names across all grouped variant files
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
