# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import sys
from abc import ABC, abstractmethod

if __package__ in (None, ""):
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tests.sweep_framework.framework.execution_capabilities import (
    is_requirement_eligible,
    requirements_from_vector_data,
    resolve_active_profile,
)
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


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
        """Find all JSON files for a given module (including grouped variants)."""
        all_files = []

        # First try exact match
        exact_match = list(self.export_dir.glob(f"{module_name}.json"))
        if exact_match:
            all_files.extend(exact_match)

        # Also look for grouped variants using the dotted suffix format.
        mesh_variants = list(self.export_dir.glob(f"{module_name}.mesh_*.json"))
        if mesh_variants:
            logger.info(f"Found {len(mesh_variants)} mesh variant file(s) for module '{module_name}'")
            all_files.extend(sorted(mesh_variants))  # Sort for consistent ordering

        hardware_variants = list(self.export_dir.glob(f"{module_name}.hw_*.json"))
        if hardware_variants:
            logger.info(f"Found {len(hardware_variants)} hardware variant file(s) for module '{module_name}'")
            all_files.extend(sorted(hardware_variants))

        if all_files:
            return sorted(set(all_files))

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

    def _get_active_profile(self):
        """Resolve the active execution capability profile for the current host."""
        try:
            profile = resolve_active_profile()
            logger.info(f"Using execution capability profile '{profile.name}' for vector filtering")
            return profile
        except RuntimeError as e:
            logger.warning(f"Execution capability profile selection skipped: {e}")
            return None

    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors from vectors_export directory (including grouped variants)."""
        module_files = self._find_module_files(module_name)
        if not module_files:
            return []

        active_profile = self._get_active_profile()

        all_vectors = []
        filtered_count = 0

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

                            if active_profile is not None:
                                requirement = requirements_from_vector_data(vector_data, module_name=module_file.stem)
                                if not is_requirement_eligible(requirement, active_profile):
                                    filtered_count += 1
                                    continue

                            all_vectors.append(vector_data)

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading vectors from {module_file}: {e}")

        if active_profile is not None and filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} vectors via execution capability profile, "
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
