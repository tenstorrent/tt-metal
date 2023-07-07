import os
from common import get_smi_log_lines, check_not_empty, get_tt_arch_from_cmd_line

def is_eth_fw_version_line(tt_arch, line):
    matches = "ETH FW Version" in line

    if matches:
        assert tt_arch != "grayskull"
        assert tt_arch in ("wormhole_b0", "wormhole", "blackhole")

    return matches

def get_fw_versions_and_dates(tt_arch):
    log_lines = get_smi_log_lines()

    version_list = []
    date_list = []

    for line in log_lines:
        occurs_version = line.find("FW Version")

        if occurs_version > -1:
            # Do not match ETH FW Version line
            if is_eth_fw_version_line(tt_arch, line):
                continue
            version_splitted = line.split(":")
            version_tmp = version_splitted[1]

            version_tmp_2 = version_tmp.split("\n")
            version_list.append(version_tmp_2[0].strip())

        occurs_date = line.find("FW Date")

        if occurs_date > -1:
            date_splitted = line.split(":")
            date_tmp = date_splitted[1]

            date_tmp_2 = date_tmp.split("\n")
            date_list.append(date_tmp_2[0].strip())

    if len(version_list) == 0 or len(date_list) == 0:
        return False, version_list, date_list, dict_out, dict_err

    return version_list, date_list


if __name__ == "__main__":
    tt_arch = get_tt_arch_from_cmd_line()

    version_list, date_list = get_fw_versions_and_dates(tt_arch)

    assert len(version_list) == len(date_list)
    assert check_not_empty(version_list)

    expected_fw_values_by_arch = {
        "grayskull": (
            ("2022-08-31", "2022-09-06"),
            ("1.0.0",),
        ),
        "wormhole_b0": (
            ("2023-03-29",),
            ("9.0.0",),
        ),
    }

    expected_tt_archs = tuple(expected_fw_values_by_arch.keys())

    assert (
        tt_arch in expected_tt_archs
    ), f"{tt_arch} not in list of supported archs: {expected_tt_archs}"

    expected_date_list = expected_fw_values_by_arch[tt_arch][0]
    expected_version_list = expected_fw_values_by_arch[tt_arch][1]

    for version, date in zip(version_list, date_list):
        assert date in expected_date_list, date_list
        assert version in expected_version_list, version_list

    print(version_list)
    print(date_list)
