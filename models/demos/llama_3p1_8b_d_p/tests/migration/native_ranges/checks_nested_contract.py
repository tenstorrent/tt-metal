"""Real import-path regression for controller/supervisor/owner contract agreement."""
import tempfile
import unittest
from pathlib import Path

import controller
import runner_support
import supervise_owner
from checks_portable_contract import make_plan


class NestedSupervisorContractTests(unittest.TestCase):
    # The three entry points must resolve the identical configured48-case contract.
    def test_actual_owner_import_path_accepts_integrated48_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            controller.validate_plan(plan)
            supervise_owner.validate_plan(plan)
            runner_support.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
