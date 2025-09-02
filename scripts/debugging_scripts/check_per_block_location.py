from dataclasses import dataclass
from decimal import Context
from typing import Callable

from block_locations_to_check import run as get_block_locations_to_check, BLOCK_TYPES, BlockLocationsToCheck
from triage import ScriptConfig, recurse_field, run_script, triage_field, triage_singleton
from ttexalens.coordinate import OnChipCoordinate

script_config = ScriptConfig(
    data_provider=True,
    depends=["block_locations_to_check"],
)


@dataclass
class PerBlockLocationCheckResult:
    location: OnChipCoordinate = triage_field("Loc")
    result: object = recurse_field()


class PerBlockLocationCheck:
    def __init__(self, block_locations_to_check: BlockLocationsToCheck):
        self.block_locations = block_locations_to_check

    def run_check(self, check: Callable[[OnChipCoordinate, str], object], block_filter: list[str] | str | None = None):
        result: list[PerBlockLocationCheckResult] = []
        block_filter = (
            BLOCK_TYPES if block_filter is None else [block_filter] if isinstance(block_filter, str) else block_filter
        )
        for device in self.block_locations.get_devices():
            for block_type in block_filter:
                for location in self.block_locations[device, block_type]:
                    check_result = check(location, block_type)
                    if check_result is None:
                        continue
                    if isinstance(check_result, list):
                        for item in check_result:
                            result.append(PerBlockLocationCheckResult(location=location, result=item))
                    else:
                        result.append(PerBlockLocationCheckResult(location=location, result=check_result))
        if len(result) == 0:
            return None
        return result


@triage_singleton
def run(args, context: Context):
    block_locations_to_check = get_block_locations_to_check(args, context)
    return PerBlockLocationCheck(block_locations_to_check)


if __name__ == "__main__":
    run_script()
