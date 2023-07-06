import os
from common import get_smi_log_lines

# Returns list of 3 parts of kernel versions (5.4.0-117-generic): ["5.4.0", 117, "generic"]
def split_kernel_version(version_string):
    splitted = version_string.split("-")

    if len(splitted) > 1:
        splitted[1] = int(splitted[1])
    else:
        splitted.append(0)

    return splitted


def get_kernel_versions():
    log_lines = get_smi_log_lines()

    kernel_ver = ""
    platform = ""
    driver_ver = ""

    for line in log_lines:
        splitted = line.split()

        if line.find("  * Kernel") > -1:
            kernel_ver = splitted[-1]

        if line.find("  * Platform") > -1:
            platform = splitted[-1]

        if line.find("  * Driver") > -1:
            driver_ver = splitted[-1]

    kernel_split = split_kernel_version(kernel_ver)

    return kernel_split, platform, driver_ver


if __name__ == "__main__":
    kernel_split, platform, driver_ver = get_kernel_versions()

    measured_version = f"{driver_ver}, {kernel_split}, {platform}"
    print(measured_version)

    assert driver_ver == "1.20.1", measured_version
