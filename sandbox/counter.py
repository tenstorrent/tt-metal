from pathlib import Path, PurePath
from functools import reduce, partial
from itertools import chain
from os import walk, chdir
from collections import namedtuple
from enum import Enum, auto
from pprint import pformat

from git import Repo, RemoteProgress, Submodule, cmd, RemoteReference
from loguru import logger
from toolz.itertoolz import first, groupby
from toolz.functoolz import compose
from toolz.dicttoolz import valmap


class Constants:
    VALID_EXTENSIONS = [
        ".bazel",
        ".c",
        ".cc",
        ".cfg",
        ".cmake",
        ".conf",
        ".config",
        ".cpp",
        ".css",
        ".def",
        ".erb",
        ".h",
        ".hpp",
        ".html",
        ".in",
        ".ini",
        ".inl",
        ".ipp",
        ".ipynb",
        ".java",
        ".js",
        ".json",
        ".la",
        ".ld",
        ".md5",
        ".mk",
        ".py",
        ".rb",
        ".sh",
        ".tcc",
        ".toml",
        ".ts",
        ".txt",
        ".yaml",
        ".yml",
        # Docs
        ".rst",
        ".md",
        ".markdown",
    ]

    GPAI_EXCLUSIONS = set(
        [
            "sandbox/",
        ]
    )

    BBE_EXCLUSIONS = set(
        [
            "ci/",
            "legacy/graph_compiler/",
        ]
    )

    PYBUDA_EXCLUSIONS = set(
        [
            "third_party/budabackend/ci/",
            "third_party/budabackend/legacy/graph_compiler",
            "ci/",
        ]
    )


class BlobClassification(Enum):
    REGULAR = auto()
    TEST = auto()


BlobInfo = namedtuple(
    "BlobInfo",
    ["path", "lines", "size_in_bytes", "classification"],
    defaults=[Path("."), 0, 0, BlobClassification.REGULAR],
)


def get_num_lines_from_rel_path(rel_path):
    try:
        return sum([1 for _ in open(rel_path)])
    except Exception as e:
        logger.critical(f"Unable to count lines in {rel_path}")
        raise e


def get_classification_from_rel_path(rel_path):
    pure_path = PurePath(rel_path)

    parts = pure_path.parts

    def part_is_test(part_):
        return part_ == "tests" or part_ == "test"

    if any(map(part_is_test, parts)):
        return BlobClassification.TEST

    return BlobClassification.REGULAR


def get_blob_info_from_rel_path(rel_path):
    assert isinstance(rel_path, Path), rel_path
    assert not rel_path.is_dir(), rel_path

    name = rel_path
    lines = get_num_lines_from_rel_path(rel_path)
    size_in_bytes = rel_path.stat().st_size
    classification = get_classification_from_rel_path(rel_path)

    return BlobInfo(name, lines, size_in_bytes, classification)


def create_repo(orig_repo_path):
    client = Repo(orig_repo_path)

    assert not client.bare

    return client


def recursively_get_all_files(dir_path):
    def get_full_paths_from_walk_result(walk_result):
        dir_name, _, filenames = walk_result

        return ((Path(dir_name) / Path(filename)) for filename in filenames)

    all_file_paths_grouped_by_dir = (
        get_full_paths_from_walk_result(walk_result) for walk_result in walk(dir_path)
    )

    all_file_paths = chain.from_iterable(all_file_paths_grouped_by_dir)

    return all_file_paths


def path_is_dir(path_):
    return path_.is_dir()


def expand_path_if_dir(path_):
    # Special case where we don't count symlinks
    is_symlink = path_.is_symlink()
    if is_symlink:
        logger.warning(f"Detected symlink {path_}")
        return []

    assert path_.exists(), f"{path_} doesn't exist in src_repo"
    if path_is_dir(path_):
        return list(
            chain.from_iterable(
                map(expand_path_if_dir, recursively_get_all_files(path_))
            )
        )
    else:
        return [path_]


expand_str_path_if_dir = compose(expand_path_if_dir, Path)

recursively_expand_rel_paths_as_strs = lambda rel_paths_: set(
    chain.from_iterable(map(expand_str_path_if_dir, rel_paths_))
)


def generate_rel_paths_of_files_to_include(
    rel_paths_as_strs,
):
    expanded_rel_paths = recursively_expand_rel_paths_as_strs(rel_paths_as_strs)

    return expanded_rel_paths


def get_rel_paths_of_files_in_repo(repo_):
    base_index_helper_entries = repo_.index.entries.values()
    get_path_from_helper_entry = lambda entry_: entry_.path
    return recursively_expand_rel_paths_as_strs(
        map(get_path_from_helper_entry, base_index_helper_entries)
    )


def get_file_extension_from_rel_path(rel_path):
    assert isinstance(rel_path, Path)
    assert (
        rel_path.exists()
    ), f"{rel_path} does not exist, while trying to find extension"
    assert not rel_path.is_dir()

    return PurePath(rel_path).suffix


def has_valid_extension(valid_extensions, rel_path):
    return get_file_extension_from_rel_path(rel_path) in valid_extensions


def rel_path_matches_all_filters(filters, rel_path):
    return all(filter_fn(rel_path) for filter_fn in filters)


def get_module_name_from_rel_path(rel_path):
    pure_path = PurePath(rel_path)

    pure_path_parts = pure_path.parts

    first_part = str(first(pure_path_parts))

    def is_legal_char(char_):
        assert isinstance(char_, str)
        assert len(char_) == 1
        return char_.isalnum() or char_ in (".", "_", "-")

    is_legal_first_part = lambda part_: all(map(is_legal_char, first_part))

    assert is_legal_first_part(first_part)

    return first_part


def turn_set_of_rel_paths_to_blob_infos(set_of_rel_paths):
    return set(map(get_blob_info_from_rel_path, set_of_rel_paths))


def add_two_blobs_unknown(a_blob, b_blob):
    assert a_blob.classification == b_blob.classification

    return BlobInfo(
        "UNKNOWN",
        a_blob.lines + b_blob.lines,
        a_blob.size_in_bytes + b_blob.size_in_bytes,
        a_blob.classification,
    )


def reduce_set_of_blob_infos(set_of_blob_infos):
    return reduce(
        add_two_blobs_unknown,
        set_of_blob_infos,
        BlobInfo("UNKNOWN", 0, 0, first(set_of_blob_infos).classification),
    )


def presentable_str_from_blob_info(blob_info):
    return f"Module with total of {blob_info.lines} lines and {blob_info.size_in_bytes} bytes"


if __name__ == "__main__":
    repo_full_path = "/home/rkim/gp.ai"
    chdir(repo_full_path)

    exclusions = Constants.GPAI_EXCLUSIONS

    repo = create_repo(repo_full_path)

    valid_extensions = Constants.VALID_EXTENSIONS

    all_repo_files = get_rel_paths_of_files_in_repo(repo)

    files_to_exclude = generate_rel_paths_of_files_to_include(exclusions)

    # Log files under track
    # logger.info(pformat(all_repo_files))
    # Log files that will be excluded
    # logger.debug(pformat(files_to_exclude))

    included_nonfiltered_files = all_repo_files - files_to_exclude

    filters = [partial(has_valid_extension, valid_extensions)]
    rel_path_matches_all_filters_ = partial(rel_path_matches_all_filters, filters)

    # Log file extensions available in repo
    # logger.trace(
    #     pformat(set(map(get_file_extension_from_rel_path, included_nonfiltered_files)))
    # )

    included_files = set(
        filter(rel_path_matches_all_filters_, included_nonfiltered_files)
    )

    included_blob_infos = set(map(get_blob_info_from_rel_path, included_files))

    logger.info("Gathered information of all matching files")

    test_blob_infos = set(
        filter(
            lambda blob_info_: blob_info_.classification == BlobClassification.TEST,
            included_blob_infos,
        )
    )
    regular_blob_infos = set(
        filter(
            lambda blob_info_: blob_info_.classification == BlobClassification.REGULAR,
            included_blob_infos,
        )
    )

    assert included_blob_infos == test_blob_infos | regular_blob_infos

    get_module_name_from_blob_info = compose(
        get_module_name_from_rel_path, lambda blob_info_: blob_info_.path
    )

    test_blob_infos_by_module = groupby(get_module_name_from_blob_info, test_blob_infos)
    regular_blob_infos_by_module = groupby(
        get_module_name_from_blob_info, regular_blob_infos
    )

    logger.info("Grouped file info by module, now aggregating statistics")

    # Log dictionary mapping of module -> blobs
    # logger.error(pformat(regular_blob_infos_by_module))

    test_blob_infos_reduced_by_module = valmap(
        reduce_set_of_blob_infos, test_blob_infos_by_module
    )
    regular_blob_infos_reduced_by_module = valmap(
        reduce_set_of_blob_infos, regular_blob_infos_by_module
    )

    logger.info("Logging regular file stats")
    logger.info(
        pformat(
            valmap(presentable_str_from_blob_info, regular_blob_infos_reduced_by_module)
        )
    )
    logger.info("Logging test file stats")
    logger.info(
        pformat(
            valmap(presentable_str_from_blob_info, test_blob_infos_reduced_by_module)
        )
    )
