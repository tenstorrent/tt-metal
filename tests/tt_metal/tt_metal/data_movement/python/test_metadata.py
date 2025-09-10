# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import yaml
from typing import Dict, Any, Tuple
import os


class TestMetadataLoader:
    def __init__(self, config):
        self.config = config

    def load_test_information(self) -> Dict[str, Any]:
        """Load test information from YAML file."""
        with open(self.config.test_info_path, "r") as file:
            return yaml.safe_load(file)

    def load_test_bounds(self) -> Dict[str, Any]:
        """Load test bounds from YAML file."""
        with open(self.config.test_bounds_path, "r") as file:
            return yaml.safe_load(file)

    def load_test_type_attributes(self) -> Dict[str, Any]:
        """Loads the test type attributes from the YAML file."""
        with open(self.config.test_type_attributes_path, "r") as file:
            return yaml.safe_load(file)

    def _get_test_id_to_name(self, test_info: Dict[str, Any]) -> Dict[str, str]:
        """Extract test_id_to_name mapping from test information."""
        return {test_id: info["name"] for test_id, info in test_info["tests"].items()}

    def _get_test_id_to_comment(self, test_info: Dict[str, Any]) -> Dict[str, str]:
        """Extract test_id_to_comment mapping from test information."""
        return {test_id: info.get("comment") for test_id, info in test_info["tests"].items() if "comment" in info}

    def _get_test_bounds(
        self, test_bounds_data: Dict[str, Any], test_id_to_name: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract test_bounds from test bounds data, converting from test names to test IDs."""
        bounds = {}

        # Create reverse mapping from test name to test ID
        name_to_test_id = {name: test_id for test_id, name in test_id_to_name.items()}

        for test_name, test_bounds in test_bounds_data.items():
            # Find the test ID for this test name
            test_id = name_to_test_id.get(test_name)
            if test_id is not None:
                for arch, arch_bounds in test_bounds.items():
                    if arch not in bounds:
                        bounds[arch] = {}
                    bounds[arch][test_id] = arch_bounds

        return bounds

    def get_test_mappings(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Get all test mappings and bounds."""
        test_info = self.load_test_information()
        test_bounds_data = self.load_test_bounds()
        test_type_attributes = self.load_test_type_attributes()

        test_id_to_name = self._get_test_id_to_name(test_info)
        test_id_to_comment = self._get_test_id_to_comment(test_info)
        test_bounds = self._get_test_bounds(test_bounds_data, test_id_to_name)

        return test_id_to_name, test_id_to_comment, test_bounds, test_type_attributes
