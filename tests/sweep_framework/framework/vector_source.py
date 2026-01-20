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

    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors from vectors_export directory (including mesh variants)"""
        module_files = self._find_module_files(module_name)
        if not module_files:
            return []

        all_vectors = []

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
                            all_vectors.append(vector_data)

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load vectors from {module_file}: {e}")
                continue

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
