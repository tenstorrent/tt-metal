# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import sys
from abc import ABC, abstractmethod

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tests.sweep_framework.framework.execution_capabilities import (
    resolve_active_profile,
)
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger
from tests.sweep_framework.framework.vector_routing import (
    ManifestFileEntry,
    is_manifest_entry_eligible,
    load_manifest_file_entries,
)


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
        self._manifest_entries_by_base: dict[str, list[ManifestFileEntry]] | None = None
        self._manifest_entries_by_module: dict[str, list[ManifestFileEntry]] | None = None
        self._manifest_path = self.export_dir / "export_manifest.json"

    def _load_manifest_index(self) -> None:
        if self._manifest_entries_by_base is not None and self._manifest_entries_by_module is not None:
            return

        self._manifest_entries_by_base = {}
        self._manifest_entries_by_module = {}

        try:
            entries = load_manifest_file_entries(self._manifest_path, repo_root=REPO_ROOT, strict=False)
        except RuntimeError as e:
            logger.warning(f"Failed to load export manifest at {self._manifest_path}: {e}")
            return

        for entry in entries:
            self._manifest_entries_by_base.setdefault(entry.base_module_name, []).append(entry)
            self._manifest_entries_by_module.setdefault(entry.module_name, []).append(entry)

    def _find_module_entries(self, module_name: str) -> list[ManifestFileEntry]:
        """Find manifest entries for a module by base name or full grouped name."""
        self._load_manifest_index()
        assert self._manifest_entries_by_base is not None
        assert self._manifest_entries_by_module is not None

        entries = list(self._manifest_entries_by_base.get(module_name, []))
        entries.extend(self._manifest_entries_by_module.get(module_name, []))
        if not entries:
            logger.warning(f"No manifest entry found for module '{module_name}' in {self._manifest_path}")
            return []

        by_path = {}
        for entry in entries:
            by_path[entry.file_path] = entry
        return sorted(by_path.values(), key=lambda item: item.file_path.name)

    def _get_active_profile(self):
        """Resolve the active execution capability profile for the current host."""
        try:
            profile = resolve_active_profile()
            logger.info(f"Using execution capability profile '{profile.name}' for vector filtering")
            return profile
        except Exception as e:
            logger.warning(f"Execution capability profile selection skipped: {e}")
            return None

    @staticmethod
    def _annotate_vector(vector_data: dict, *, input_hash: str, suite_name: str, module_name: str) -> None:
        """Attach common vector metadata used by the runner."""
        vector_data["input_hash"] = input_hash
        vector_data["suite_name"] = suite_name
        # Preserve stored sweep_name (may include mesh suffix), fallback to module_name
        if "sweep_name" not in vector_data:
            vector_data["sweep_name"] = module_name

    def load_vectors(self, module_name: str, suite_name: str | None = None, vector_id: str | None = None) -> list[dict]:
        """Load test vectors from manifest-declared vectors_export files."""
        module_entries = self._find_module_entries(module_name)
        if not module_entries:
            return []

        # Explicit vector lookup should bypass eligibility filtering.
        active_profile = None if vector_id else self._get_active_profile()
        if active_profile is not None:
            eligible_entries = [
                entry
                for entry in module_entries
                if is_manifest_entry_eligible(
                    grouping_kind=entry.grouping_kind,
                    hardware_group=entry.hardware_group,
                    mesh_shapes=entry.mesh_shapes,
                    profile=active_profile,
                )
            ]
            skipped_entries = len(module_entries) - len(eligible_entries)
            if skipped_entries > 0:
                logger.info(
                    f"Skipped {skipped_entries} manifest vector file(s) via execution capability profile, "
                    f"selected {len(eligible_entries)}"
                )
            module_entries = eligible_entries

        all_vectors = []

        # Load vectors from all matching files (e.g., base + mesh variants)
        for entry in module_entries:
            module_file = entry.file_path
            try:
                with open(module_file, "r") as file:
                    data = json.load(file)

                for suite_key, suite_content in data.items():
                    if entry.suite_names and suite_key not in entry.suite_names:
                        continue
                    if suite_name and suite_name != suite_key:
                        continue

                    if vector_id:
                        if vector_id in suite_content:
                            vector = suite_content[vector_id]
                            self._annotate_vector(
                                vector,
                                input_hash=vector_id,
                                suite_name=suite_key,
                                module_name=module_name,
                            )
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
                            self._annotate_vector(
                                vector_data,
                                input_hash=input_hash,
                                suite_name=suite_key,
                                module_name=module_name,
                            )
                            all_vectors.append(vector_data)

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading vectors from {module_file}: {e}")

        return all_vectors

    def get_available_suites(self, module_name: str) -> list[str]:
        """Get list of available suites for a module from manifest-declared vector files."""
        module_entries = self._find_module_entries(module_name)
        if not module_entries:
            return []

        # Collect unique suite names across all grouped variant files
        all_suites = set()
        for entry in module_entries:
            module_file = entry.file_path
            if entry.suite_names:
                all_suites.update(entry.suite_names)
                continue
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
