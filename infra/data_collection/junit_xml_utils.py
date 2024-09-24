# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from loguru import logger
from defusedxml.ElementTree import parse as XMLParse
from toolz.dicttoolz import merge


def get_xml_file_root_element_tree(filepath):
    root_element_tree = XMLParse(filepath)
    root_element = root_element_tree.getroot()

    assert root_element.tag == "testsuites"

    return root_element_tree


def sanity_check_pytest_junit_xml_(root_element):
    testsuite_count = len(root_element)

    assert testsuite_count == 1, f"{len(root_element)}"

    logger.debug("Asserted pytest junit xml")

    return root_element


def is_pytest_junit_xml(root_element):
    is_pytest = root_element[0].get("name") == "pytest"

    if is_pytest:
        sanity_check_pytest_junit_xml_(root_element)

    return is_pytest


def get_at_most_one_single_child_element_(element, tag_name):
    is_expected = lambda child_: child_.tag == tag_name

    potential_expected_blocks = list(filter(is_expected, element))

    assert (
        len(potential_expected_blocks) <= 1
    ), f"{len(potential_expected_blocks)} is not exactly 1 for tag name {tag_name}"

    return potential_expected_blocks[0] if len(potential_expected_blocks) else None


def get_pytest_testcase_properties(testcase_element):
    properties_block = get_at_most_one_single_child_element_(testcase_element, "properties")

    assert properties_block is not None

    def get_property_as_dict_(property_):
        assert property_.tag == "property"

        return dict([(property_.attrib["name"], property_.attrib["value"])])

    return reduce(merge, map(get_property_as_dict_, properties_block), {})


def get_optional_child_element_exists_(parent_element, tag_name):
    return get_at_most_one_single_child_element_(parent_element, tag_name) != None


def get_pytest_testcase_is_skipped(testcase_element):
    return get_optional_child_element_exists_(testcase_element, "skipped")


def get_pytest_testcase_is_failed(testcase_element):
    return get_optional_child_element_exists_(testcase_element, "failure")


def get_pytest_testcase_is_error(testcase_element):
    return get_optional_child_element_exists_(testcase_element, "error")


# opportunity for less copy-pasta


def get_pytest_failure_message(testcase_element):
    assert get_pytest_testcase_is_failed(testcase_element)

    failure_element = get_at_most_one_single_child_element_(testcase_element, "failure")

    return failure_element.attrib["message"]


def get_pytest_error_message(testcase_element):
    assert get_pytest_testcase_is_error(testcase_element)

    error_element = get_at_most_one_single_child_element_(testcase_element, "error")

    return error_element.attrib["message"]
