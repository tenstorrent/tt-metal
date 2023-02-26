from os import walk, environ
from shutil import copy
from pathlib import Path
from pprint import pformat
from operator import contains, getitem, eq
from functools import partial
from itertools import filterfalse, chain

from loguru import logger
from git import Repo, RemoteProgress, Submodule, cmd, RemoteReference
from git.refs.symbolic import SymbolicReference
from github import Github
from toolz.itertoolz import groupby, first
from toolz.dicttoolz import valmap
from toolz.functoolz import compose

from reg_scripts import common


class ReleaseConstants:
    PUBLIC_GITHUB_USER_HOST = "git@github.com"

    REMOTE_ORGANIZATION = "tenstorrent-accel"

    REMOTE_RELEASE_REPO = "gp.ai-rel-test"

    REMOTE_RELEASE_BRANCH = "main"

    RELEASE_EXCLUSIONS = tuple(
        map(
            Path,
            [
                "sandbox/",
                "compiler/tests/",
                "llrt/tests/",
                "ll_buda/tests/",
                "build_kernels_for_riscv/tests/",
                "gpai/tests/",
                "release/",
                "reg_scripts/",
                "kernels/compute/draft/",
                ".gitmodules",  # needed for submodules
                "python_api_testing/",
            ],
        )
    )

    DST_DELETION_EXCEPTIONS_PATHS = tuple(
        map(
            Path,
            [
                ".gitmodules",
            ],
        )
    )


def create_remote_release_repo_url(user_host, owner, repo):
    return f"{user_host}:{owner}/{repo}.git"


def create_orig_repo_client(orig_repo_path):
    client = Repo(orig_repo_path)

    assert not client.bare

    return client


def create_cloned_release_repo_client(remote_release_repo_url, to_path, progress):
    """
    Assumes SSH access available in path and main branch created on remote, bug
    in this repo
    """
    multi_options = ["--recurse-submodules"]

    client = Repo.clone_from(
        url=remote_release_repo_url,
        to_path=to_path,
        progress=progress,
        multi_options=multi_options,
    )

    if client.bare:
        logger.warning("Release repo remote was empty")

    is_new_repo_on_gh = len(client.heads) == 0
    if is_new_repo_on_gh:
        logger.info("New repo detected, creating main branch")
        git_cmd = cmd.Git(to_path)
        git_cmd.execute("git checkout -b main".split(" "))
        assert not client.head.is_detached

    return client


def attempt_fetch_and_pull_origin_for_repo(repo, head_to_checkout, progress):
    remote = repo.remote(name="origin")

    fetch_results = remote.fetch(progress=progress)

    for fetch_result in fetch_results:
        logger.info(f"Remote fetch result: {fetch_result.flags}")

    head_to_checkout.checkout()

    pull_results = remote.pull()

    for pull_result in pull_results:
        logger.info(f"Remote pull result: {pull_result.flags}")


def checkout_repo_to_specific_commit(repo, head_to_checkout, commit, progress):
    attempt_fetch_and_pull_origin_for_repo(repo, head_to_checkout, progress)

    logger.info(commit)

    repo.head.reset(commit, working_tree=True)

    return repo


def create_remote_progress_printer():
    class ProgressPrinter(RemoteProgress):
        def update(self, op_code, cur_count, max_count=None, message=""):
            max_count = max_count or 100
            message = message or "No message"

            logger.info(f"  [{op_code}]: {cur_count} / {max_count} | {message}")

    return ProgressPrinter()


def copy_submodule_config_from_repo(src_repo, dst_repo, progress):
    """
    We have two steps to do:
    - Add any new submodules
    - Update any existing ones to the same commit as the other

    BIG NOTES:
    - Assume no submodules will be deleted, may have to address in future
    - Not added to working index, must do later as large commit
    """

    src_repo_submodules = tuple(src_repo.submodules)
    dst_repo_submodules = tuple(dst_repo.submodules)

    get_submodule_name = lambda sm_: sm_.name
    existing_submodules_by_name_in_dst = valmap(
        first, groupby(get_submodule_name, dst_repo_submodules)
    )
    submodules_by_name_in_src = valmap(
        first, groupby(get_submodule_name, src_repo_submodules)
    )

    submodule_name_exists_in_dst = partial(contains, existing_submodules_by_name_in_dst)
    submodule_exists_in_dst = compose(submodule_name_exists_in_dst, get_submodule_name)

    submodules_not_in_dst = set(
        filterfalse(submodule_exists_in_dst, src_repo_submodules)
    )

    src_submodules_in_dst = set(src_repo_submodules) - submodules_not_in_dst

    get_dst_submodule_by_name = partial(getitem, existing_submodules_by_name_in_dst)
    get_dst_submodule_by_src_submodule = compose(
        get_dst_submodule_by_name, get_submodule_name
    )

    def get_commit_of_head_of_sm(sm):
        return sm.module().head.commit

    def create_sm_with_new_repo_commit(sm, commit):
        sm.binsha = commit.binsha
        return sm

    def update_dst_sm_to_match_src_sm(dst_sm, src_sm):
        commit = get_commit_of_head_of_sm(src_sm)
        dst_sm_repo = dst_sm.module()
        dst_master_head = first(dst_sm_repo.heads)
        dst_sm_repo = checkout_repo_to_specific_commit(
            dst_sm_repo, dst_master_head, commit, progress
        )
        dst_sm = create_sm_with_new_repo_commit(dst_sm, commit)
        return dst_sm

    def add_new_submodule(src_submodule_to_add):
        new_submodule_no_update = Submodule.add(
            dst_repo,
            src_submodule_to_add.name,
            src_submodule_to_add.path,
            src_submodule_to_add.url,
        )
        new_submodule = update_dst_sm_to_match_src_sm(
            new_submodule_no_update, src_submodule_to_add
        )

        logger.info(f"Added submodule {new_submodule.name} to repo")

        return new_submodule

    # Add new submodules
    new_submodules = set(map(add_new_submodule, submodules_not_in_dst))

    def update_existing_submodule(src_submodule_to_update_in_dst):
        dst_submodule_to_update = get_dst_submodule_by_src_submodule(
            src_submodule_to_update_in_dst
        )
        updated_dst_submodule = update_dst_sm_to_match_src_sm(
            dst_submodule_to_update, src_submodule_to_update_in_dst
        )

        logger.info(f"Updated submodule {updated_dst_submodule.name} to repo")

        return updated_dst_submodule

    # Update any existing submodules to current commit
    updated_submodules = set(map(update_existing_submodule, src_submodules_in_dst))

    return dst_repo, new_submodules | updated_submodules


def get_tracked_files_under_repo(repo_):
    base_index_helper_entries = repo_.index.entries.values()
    get_path_from_helper_entry = lambda entry_: entry_.path
    get_path_from_helper_entry_as_path = compose(Path, get_path_from_helper_entry)
    return set(map(get_path_from_helper_entry_as_path, base_index_helper_entries))


def get_rel_path_to_submodule(sm_):
    sm_path = Path(sm_.path)

    assert sm_path.exists()
    assert sm_path.is_dir()

    return sm_path


def generate_rel_paths_of_files_to_copy(
    src_repo,
    dst_repo,
    exclusions_rel_paths_as_strs,
):
    def get_full_paths_from_walk_result(walk_result):
        dir_name, _, filenames = walk_result

        return ((Path(dir_name) / Path(filename)) for filename in filenames)

    def recursively_get_all_files(dir_path):
        all_file_paths_grouped_by_dir = (
            get_full_paths_from_walk_result(walk_result)
            for walk_result in walk(dir_path)
        )

        all_file_paths = chain.from_iterable(all_file_paths_grouped_by_dir)

        return all_file_paths

    def path_is_dir(path_):
        return path_.is_dir()

    def expand_path_if_dir(path_):
        assert path_.exists(), f"{path_} doesn't exist in src_repo"
        if path_is_dir(path_):
            return recursively_get_all_files(path_)
        else:
            return [path_]

    expand_str_path_if_dir = compose(expand_path_if_dir, Path)

    expanded_exclusions_rel_paths_no_sms = set(
        chain.from_iterable(map(expand_str_path_if_dir, exclusions_rel_paths_as_strs))
    )

    src_submodules = set(src_repo.submodules)

    submodule_rel_paths = set(map(get_rel_path_to_submodule, src_submodules))

    # Get sm paths themselves + all they contain into one cohesive set
    submodules_contents_expanded_no_top_level = set(
        chain.from_iterable(map(expand_path_if_dir, submodule_rel_paths))
    )
    submodules_expanded_rel_paths = (
        submodule_rel_paths | submodules_contents_expanded_no_top_level
    )

    expanded_exclusions_rel_paths = (
        submodules_expanded_rel_paths | expanded_exclusions_rel_paths_no_sms
    )

    possible_rel_paths_to_include = get_tracked_files_under_repo(src_repo)

    rel_paths_of_files_to_copy = (
        possible_rel_paths_to_include - expanded_exclusions_rel_paths
    )

    return rel_paths_of_files_to_copy


def generate_rel_paths_of_non_sm_files_to_delete_in_dst(
    src_repo,
    dst_repo,
    rel_paths_of_files_to_copy_in_src,
    deletion_exceptions_paths,
):
    rel_paths_in_dst = get_tracked_files_under_repo(dst_repo)

    assert isinstance(
        rel_paths_of_files_to_copy_in_src, set
    ), f"Rel paths of files must be given as a set"

    # Get submodules of both, since we may have copied over SMs that
    # don't exist yet in dst
    # Submodule reconciliation is done separately
    dst_repo_submodules = dst_repo.submodules

    dst_repo_sm_rel_paths = set(map(get_rel_path_to_submodule, dst_repo_submodules))

    src_repo_submodules = src_repo.submodules

    src_repo_sm_rel_paths = set(map(get_rel_path_to_submodule, src_repo_submodules))

    return (
        rel_paths_in_dst
        - rel_paths_of_files_to_copy_in_src
        - dst_repo_sm_rel_paths
        - src_repo_sm_rel_paths
        - deletion_exceptions_paths
    )


def copy_reg_files_from_src_to_dst_repo(
    src_repo,
    dst_repo,
    src_repo_abs_path,
    dst_repo_abs_path,
    rel_paths_of_files_to_copy,
    progress,
):
    for rel_path in rel_paths_of_files_to_copy:
        src_path = src_repo_abs_path / rel_path
        assert src_path.exists(), f"{src_path} does not exist"
        assert not src_path.is_dir(), f"{src_path} is not a file"

    total_files_count = len(rel_paths_of_files_to_copy)
    for file_idx, rel_path_to_copy in enumerate(rel_paths_of_files_to_copy):
        src_abs_path = src_repo_abs_path / rel_path_to_copy
        dst_abs_path = dst_repo_abs_path / rel_path_to_copy

        dst_path_parent = dst_abs_path.parent
        if not dst_path_parent.exists():
            dst_path_parent.mkdir(parents=True)

        copy(src_abs_path, dst_abs_path)
        file_num = file_idx + 1
        if not file_num % 100:
            logger.info(f"  File copy - {file_num} / {total_files_count} completed")

    return dst_repo


def delete_reg_files_in_folder(
    folder_root_abs_path,
    rel_paths_files_to_delete,
):
    create_abs_path_from_rel_path = lambda rel_path_: folder_root_abs_path / rel_path_

    abs_paths = set(map(create_abs_path_from_rel_path, rel_paths_files_to_delete))

    for abs_path in abs_paths:
        assert abs_path.exists(), f"{abs_path} marked for deletion, but does not exist"
        assert (
            not abs_path.is_dir()
        ), f"{abs_path} is a dir, but marked for deletion. We only delete files. Is it a submodule?"

        abs_path.unlink()

    return abs_paths


def stage_to_commit_everything_in_repo(
    repo,
    submodules_to_add,
    blobs_and_files_to_add,
    rel_paths_strs_to_delete,
    dst_repo_path,
    force=False,
):
    items_to_add = list(submodules_to_add) + list(blobs_and_files_to_add)

    current_index = repo.index

    added_entries = current_index.add(
        items_to_add,
        force=force,
    )

    deleted_entries = (
        current_index.remove(rel_paths_strs_to_delete)
        if rel_paths_strs_to_delete
        else []
    )

    # ugh I hate this... just for device
    # Should assert somewhere that lfs files are only in device
    git_cmd = cmd.Git(dst_repo_path)
    git_cmd.execute("git add device".split(" "))

    logger.info(f"Added submodules and everything trackable in {repo.working_dir}")

    return repo


def assert_unstaged_diff_is_nonempty(repo):
    has_contents = repo.is_dirty(untracked_files=True)

    assert (
        has_contents
    ), f"Empty diff - looks like no difference between last release and this"

    return repo


def commit_staged_diffs_in_repo(repo, commit_msg):
    current_index = repo.index

    new_commit = current_index.commit(commit_msg)
    assert new_commit == repo.head.commit

    logger.info(f"Commit created with message: {commit_msg}")

    return repo


def push_current_repo(repo, remote_name, progress):
    remote = repo.remote(name="origin")

    main_branch = first(repo.heads)

    has_remote_branch = main_branch.tracking_branch()

    if not has_remote_branch:
        remote_ref = RemoteReference(repo, f"refs/remotes/origin/main")
        repo.head.reference.set_tracking_branch(remote_ref)

    push_results = remote.push(progress=progress)

    for push_result in push_results:
        logger.info(f"Remote push result: {push_result.flags}")

    return repo


def assert_that_release_name_is_unique(github_release_repo, release_name):
    get_release_name_from_release = lambda release_: release_.title

    title_matches_proposed_name = partial(eq, release_name)

    release_matches_proposed_name = compose(
        title_matches_proposed_name,
        str,
        get_release_name_from_release,
    )

    matching_releases = tuple(
        filter(release_matches_proposed_name, github_release_repo.get_releases())
    )

    assert (
        not matching_releases
    ), f"A release with the name {release_name} already exists"


def get_commit_sha_of_head_of_repo(repo):
    return str(repo.commit(ReleaseConstants.REMOTE_RELEASE_BRANCH).hexsha)


def delete_everything_in_path(path_to_delete):
    assert path_to_delete.exists()

    delete_process = common.run_process_and_get_result(f"rm -rf {path_to_delete}")

    assert common.completed_process_passed(
        delete_process
    ), f"Delete process failed for {path_to_delete}"


def create_public_github_client(github_pat):
    return Github(github_pat)


if __name__ == "__main__":
    github_pat = environ.get("GITHUB_PAT", "")

    if not github_pat:
        raise Exception("You must provide a GITHUB_PAT personal access token")

    orig_repo_path = Path(common.get_git_home_dir_str())
    remote_release_repo_url = create_remote_release_repo_url(
        ReleaseConstants.PUBLIC_GITHUB_USER_HOST,
        ReleaseConstants.REMOTE_ORGANIZATION,
        ReleaseConstants.REMOTE_RELEASE_REPO,
    )

    release_repo_path = orig_repo_path / "release/release_build"

    if release_repo_path.exists():
        logger.info("Deleting existing release build folder")
        delete_everything_in_path(release_repo_path)

    orig_repo_client = create_orig_repo_client(orig_repo_path)

    progress_printer = create_remote_progress_printer()

    release_repo_client = create_cloned_release_repo_client(
        remote_release_repo_url, release_repo_path, progress_printer
    )
    assert release_repo_path.exists()
    assert release_repo_path.is_dir()

    logger.info(f"Release repo cloned into {release_repo_path}")

    release_exclusions = ReleaseConstants.RELEASE_EXCLUSIONS

    release_repo_client, dst_submodules = copy_submodule_config_from_repo(
        orig_repo_client, release_repo_client, progress_printer
    )

    rel_paths_of_files_to_copy = generate_rel_paths_of_files_to_copy(
        orig_repo_client,
        release_repo_client,
        release_exclusions,
    )

    release_repo_client = copy_reg_files_from_src_to_dst_repo(
        orig_repo_client,
        release_repo_client,
        orig_repo_path,
        release_repo_path,
        rel_paths_of_files_to_copy,
        progress_printer,
    )

    deletion_exceptions_paths = set(ReleaseConstants.DST_DELETION_EXCEPTIONS_PATHS)
    rel_paths_of_non_sm_files_to_delete_in_dst = (
        generate_rel_paths_of_non_sm_files_to_delete_in_dst(
            orig_repo_client,
            release_repo_client,
            rel_paths_of_files_to_copy,
            deletion_exceptions_paths,
        )
    )

    if rel_paths_of_non_sm_files_to_delete_in_dst:
        logger.warning("Files have been marked for deletion in dst repo")
        logger.error(pformat(rel_paths_of_non_sm_files_to_delete_in_dst))

    commit_msg = "release"
    remote_name = "origin"

    release_repo_client = assert_unstaged_diff_is_nonempty(release_repo_client)

    rel_paths_strs_of_non_sm_files_to_delete_in_dst = set(
        map(str, rel_paths_of_non_sm_files_to_delete_in_dst)
    )

    release_repo_client = stage_to_commit_everything_in_repo(
        release_repo_client,
        dst_submodules,
        rel_paths_of_files_to_copy,
        rel_paths_strs_of_non_sm_files_to_delete_in_dst,
        release_repo_path,
    )

    abs_paths_deleted_files = delete_reg_files_in_folder(
        release_repo_path,
        rel_paths_of_non_sm_files_to_delete_in_dst,
    )

    release_repo_client = commit_staged_diffs_in_repo(release_repo_client, commit_msg)

    github_client = create_public_github_client(github_pat)

    github_release_org = github_client.get_organization(
        ReleaseConstants.REMOTE_ORGANIZATION
    )
    github_release_repo = github_release_org.get_repo(
        ReleaseConstants.REMOTE_RELEASE_REPO
    )

    release_name = "vpre-alpha1"

    assert_that_release_name_is_unique(github_release_repo, release_name)

    release_repo_client = push_current_repo(
        release_repo_client, remote_name, progress_printer
    )

    release_message = "Release notes pending."

    tag = release_name
    tag_message = "automated tag release"

    tag_object = get_commit_sha_of_head_of_repo(release_repo_client)
    logger.info(f"Creating a release on commit {tag_object}")

    tag_type = "commit"
    github_release = github_release_repo.create_git_tag_and_release(
        tag,
        tag_message,
        release_name,
        release_message,
        tag_object,
        tag_type,
    )

    release_info = {"release_commit": tag_object, "release_name": release_name}

    logger.info("Created release. Printing info:")
    logger.info(pformat(release_info))
