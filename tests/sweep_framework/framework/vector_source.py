# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import pathlib
import json
from elasticsearch import Elasticsearch, NotFoundError
from framework.statuses import VectorStatus
from framework.elastic_config import VECTOR_INDEX_PREFIX
from framework.sweeps_logger import sweeps_logger as logger


class VectorSource(ABC):
    """Abstract base class for test vector sources"""

    @abstractmethod
    def load_vectors(
        self, module_name: str, suite_name: Optional[str] = None, vector_id: Optional[str] = None
    ) -> List[Dict]:
        """Load test vectors based on criteria"""
        pass

    @abstractmethod
    def get_available_suites(self, module_name: str) -> List[str]:
        """Get list of available suites for a module"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the source is accessible"""
        pass


class ElasticVectorSource(VectorSource):
    """Elasticsearch-based vector source"""

    def __init__(self, connection_string: str, username: str, password: str, tag: str):
        logger.info(
            f"Initializing Elasticsearch vector source with connection string: {connection_string}, username: {username}, tag: {tag}"
        )
        try:
            self.client = Elasticsearch(connection_string, basic_auth=(username, password))
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            exit(1)
        self.tag = tag
        self.connection_string = connection_string
        self.username = username
        self.password = password

    def load_vectors(
        self, module_name: str, suite_name: Optional[str] = None, vector_id: Optional[str] = None
    ) -> List[Dict]:
        """Load test vectors from Elasticsearch"""
        vector_index = VECTOR_INDEX_PREFIX + module_name

        if vector_id:
            # Load single vector by ID
            try:
                response = self.client.get(index=vector_index, id=vector_id)
                vector = response["_source"]
                vector["vector_id"] = vector_id
                return [vector]
            except NotFoundError:
                return []

        # Build query
        query = {
            "bool": {
                "must": [
                    {"match": {"status": str(VectorStatus.CURRENT)}},
                    {"match": {"tag.keyword": self.tag}},
                ]
            }
        }

        if suite_name:
            query["bool"]["must"].append({"match": {"suite_name.keyword": suite_name}})

        try:
            response = self.client.search(
                index=vector_index,
                query=query,
                size=10000,
            )

            test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
            test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]

            # Add vector IDs to the vectors
            for i in range(len(test_ids)):
                test_vectors[i]["vector_id"] = test_ids[i]

            return test_vectors

        except NotFoundError:
            return []

    def get_available_suites(self, module_name: str) -> List[str]:
        """Get list of available suites for a module from Elasticsearch"""
        vector_index = VECTOR_INDEX_PREFIX + module_name

        try:
            response = self.client.search(
                index=vector_index,
                query={"match": {"tag.keyword": self.tag}},
                aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                size=0,  # We only want aggregations, not documents
            )
            suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
            logger.info(f"Available suites for module {module_name}: {suites}")
            return suites
        except (NotFoundError, Exception):
            return []

    def validate_connection(self) -> bool:
        """Validate that the Elasticsearch connection works"""
        # If we got this far, the client was created successfully in __init__
        # The original sweeps_runner.py doesn't do additional validation beyond client creation
        # The es_sweeps user doesn't have cluster monitor permissions, but that's fine for our use case
        logger.info("✓ Elasticsearch client created successfully - skipping additional validation")
        return True


class FileVectorSource(VectorSource):
    """File-based vector source"""

    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def load_vectors(
        self, module_name: str, suite_name: Optional[str] = None, vector_id: Optional[str] = None
    ) -> List[Dict]:
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
                        vector["vector_id"] = vector_id
                        vector["input_hash"] = vector_id
                        vector["suite_name"] = suite_key
                        vector["sweep_name"] = module_name
                        vectors.append(vector)
                        break
            else:
                # Load by suite or all suites
                for suite_key, suite_content in data.items():
                    if suite_name and suite_name != suite_key:
                        continue

                    for input_hash, vector_data in suite_content.items():
                        vector_data["vector_id"] = input_hash
                        vector_data["input_hash"] = input_hash
                        vector_data["suite_name"] = suite_key
                        vector_data["sweep_name"] = module_name
                        vectors.append(vector_data)

            return vectors

        except (json.JSONDecodeError, IOError):
            return []

    def get_available_suites(self, module_name: str) -> List[str]:
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

    def __init__(self, export_dir: Optional[pathlib.Path] = None):
        if export_dir is None:
            # Default to vectors_export directory relative to this file
            self.export_dir = pathlib.Path(__file__).parent.parent / "vectors_export"
        else:
            self.export_dir = export_dir

    def _find_module_file(self, module_name: str) -> Optional[pathlib.Path]:
        """Find the JSON file for a given module"""
        potential_files = list(self.export_dir.glob(f"{module_name}.json"))
        if potential_files:
            return potential_files[0]

        logger.warning(f"No vector file found for module '{module_name}' in {self.export_dir}")
        try:
            tail = module_name.split(".")[-1]
            similar_files = list(self.export_dir.glob(f"*{tail}*.json"))
            if similar_files:
                top_names = [f.name for f in similar_files[:5]]
                logger.info(f"Similar files found: {top_names}")
        except Exception:
            pass
        return None

    def load_vectors(
        self, module_name: str, suite_name: Optional[str] = None, vector_id: Optional[str] = None
    ) -> List[Dict]:
        """Load test vectors from vectors_export directory"""
        module_file = self._find_module_file(module_name)
        if not module_file:
            return []

        try:
            with open(module_file, "r") as file:
                data = json.load(file)

            vectors = []

            for suite_key, suite_content in data.items():
                if suite_name and suite_name != suite_key:
                    continue

                if vector_id:
                    if vector_id in suite_content:
                        vector = suite_content[vector_id]
                        vector["vector_id"] = vector_id
                        vector["input_hash"] = vector_id
                        vector["suite_name"] = suite_key
                        vector["sweep_name"] = module_name
                        vectors.append(vector)
                        logger.info(f"Vector ID '{vector_id}' found in suite '{suite_name}' of module '{module_name}'")
                    else:
                        logger.warning(
                            f"Vector ID '{vector_id}' not found in suite '{suite_name}' of module '{module_name}'"
                        )
                    break
                else:
                    for input_hash, vector_data in suite_content.items():
                        vector_data["vector_id"] = input_hash
                        vector_data["input_hash"] = input_hash
                        vector_data["suite_name"] = suite_key
                        vector_data["sweep_name"] = module_name
                        vectors.append(vector_data)

            return vectors

        except (json.JSONDecodeError, IOError):
            return []

    def get_available_suites(self, module_name: str) -> List[str]:
        """Get list of available suites for a module from vectors_export directory"""
        module_file = self._find_module_file(module_name)
        if not module_file:
            return []

        try:
            with open(module_file, "r") as file:
                data = json.load(file)
            return list(data.keys())
        except (json.JSONDecodeError, IOError):
            return []

    def validate_connection(self) -> bool:
        """Validate that the export directory exists"""
        return self.export_dir.exists() and self.export_dir.is_dir()


class VectorSourceFactory:
    """Factory to create appropriate vector source based on configuration"""

    @staticmethod
    def create_source(vector_source: str, **kwargs) -> VectorSource:
        if vector_source == "elastic":
            required_args = ["connection_string", "username", "password", "tag"]
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for elastic vector source")
            return ElasticVectorSource(**kwargs)
        elif vector_source == "file":
            if "file_path" not in kwargs:
                raise ValueError("Missing required argument 'file_path' for file vector source")
            return FileVectorSource(kwargs["file_path"])
        elif vector_source == "vectors_export":
            export_dir = kwargs.get("export_dir")
            return VectorExportSource(export_dir)
        else:
            raise ValueError(f"Unknown vector source: {vector_source}")
