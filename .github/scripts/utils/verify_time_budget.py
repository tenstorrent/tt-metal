import argparse
import yaml
import sys
from collections import defaultdict


def verify_timeouts(tests_file, time_budget_file, workflow_name, tier=None, max_per_test_timeout=None):
    """
    Verifies that the SUM of all test timeouts for each (Team, SKU) pair in tests_file
    is within the total time budget defined in time_budget_file for the given workflow.

    When `tier` is provided, only the SKU entries whose `tier` matches are summed, and
    the budget is looked up under the per-tier key "<workflow_name>_tier<tier>" (e.g.
    "unit_tier1"). When `tier` is None the behaviour is unchanged: every SKU entry is
    summed and the budget is looked up under the plain "<workflow_name>" key.

    When `max_per_test_timeout` is provided, every individual SKU timeout in the tests
    file must be <= that limit (minutes). Used by smoke/basic to enforce a per-entry
    ceiling for a given pipeline (e.g. merge_gate).
    """
    budget_workflow = workflow_name if tier is None else f"{workflow_name}_tier{tier}"

    print(f"Loading time budgets from: {time_budget_file}")
    with open(time_budget_file, "r") as f:
        budgets = yaml.safe_load(f)

    print(f"Loading tests from: {tests_file}")
    with open(tests_file, "r") as f:
        tests = yaml.safe_load(f) or []

    if tier is not None:
        print(f"Filtering tests to tier '{tier}'; budgets looked up under workflow key '{budget_workflow}'.")

    if max_per_test_timeout is not None:
        print(f"Enforcing max per-test timeout of {max_per_test_timeout} min " f"for pipeline '{workflow_name}'.")

    errors_found = False

    # --- Part 1: Validate test file format and sum timeouts per (Team, SKU) pair ---
    print("\n--- Summing Test Timeouts per Team and SKU ---")

    # The key will now be a tuple: (team, sku)
    # e.g., {('ops', 'N300'): 20, ('models', 'N300'): 15}
    budget_totals = defaultdict(int)

    for test in tests:
        test_name = test.get("name", "Unnamed Test")

        # Validate that all mandatory keys exist for this test
        required_keys = ["skus", "team"]
        missing_keys = [key for key in required_keys if key not in test]
        if missing_keys:
            print(
                f"  [ERROR] Validation FAILED! Test '{test_name}' is missing mandatory keys: {', '.join(missing_keys)}."
            )
            errors_found = True
            continue  # Skip this invalid test

        test_skus = test["skus"]
        test_team = test["team"]

        if not isinstance(test_skus, dict) or not test_skus:
            print(
                f"  [ERROR] Validation FAILED! Test '{test_name}' has invalid 'skus' field. "
                f"Expected a non-empty mapping of SKU names to their config."
            )
            errors_found = True
            continue

        for sku_name, sku_config in test_skus.items():
            if not isinstance(sku_config, dict) or "timeout" not in sku_config:
                print(f"  [ERROR] Validation FAILED! Test '{test_name}', SKU '{sku_name}' is missing 'timeout'.")
                errors_found = True
                continue

            # When verifying a specific tier, only count SKU entries belonging to it.
            if tier is not None and str(sku_config.get("tier")) != str(tier):
                continue

            test_timeout = sku_config["timeout"]

            if max_per_test_timeout is not None and float(test_timeout) > float(max_per_test_timeout):
                print(
                    f"  [ERROR] Per-test timeout FAILED! Test '{test_name}', SKU '{sku_name}' "
                    f"has timeout {test_timeout} min, which exceeds the max per-test timeout of "
                    f"{max_per_test_timeout} min for pipeline '{workflow_name}'."
                )
                errors_found = True

            # Use a tuple (team, sku) as the key for summation
            budget_key = (test_team, sku_name)
            budget_totals[budget_key] += test_timeout
            print(f"  Test '{test_name}' (Team: {test_team}, SKU: {sku_name}) adds {test_timeout} min.")

    if errors_found:
        print(f"\nValidation errors in {tests_file}. Please fix the entries above.")
        sys.exit(1)

    # --- Part 2: Verify Total Time Budget for each (Team, SKU) pair ---
    print("\n--- Verifying Total Time Budgets ---")

    for (team, sku), total_time_requested in budget_totals.items():
        try:
            # Navigate the budgets config using team, workflow (or per-tier key), and sku
            total_time_budget = budgets[team][budget_workflow][sku]

            print(
                f"Checking total for Team '{team}', SKU '{sku}': Requested = {total_time_requested} min, Budget = {total_time_budget} min"
            )

            if total_time_requested > total_time_budget:
                print(
                    f"  [ERROR] Total Time Budget FAILED for Team '{team}', SKU '{sku}'! "
                    f"The sum of test timeouts ({total_time_requested} min) exceeds the allocated budget of {total_time_budget} min."
                )
                errors_found = True
            else:
                print(f"  [OK] Total time for Team '{team}', SKU '{sku}' is within the budget.")

        except (KeyError, TypeError):
            print(
                f"  [ERROR] Configuration FAILED! Could not find a 'time_budget' for Team '{team}', Workflow '{budget_workflow}', SKU '{sku}' in {time_budget_file}."
            )
            errors_found = True
            continue

    # --- Final Verdict ---
    if errors_found:
        print("\nVerification failed.")
        sys.exit(1)
    else:
        print("\nVerification successful.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify test yaml timeouts against team time budgets (and optional per-test ceiling)."
    )
    parser.add_argument("tests_file", help="Path to pipeline_reorg tests yaml")
    parser.add_argument("time_budget_file", help="Path to time_budget.yaml")
    parser.add_argument("workflow_name", help="Budget workflow key (e.g. merge_gate, pr_gate, unit)")
    parser.add_argument(
        "tier",
        nargs="?",
        default=None,
        help="Optional tier; budgets looked up under <workflow_name>_tier<tier>",
    )
    parser.add_argument(
        "--max-per-test-timeout",
        type=float,
        default=None,
        help="Optional max allowed timeout (minutes) for any individual test SKU entry",
    )
    args = parser.parse_args()

    tier_arg = args.tier.strip() if args.tier and args.tier.strip() else None
    verify_timeouts(args.tests_file, args.time_budget_file, args.workflow_name, tier_arg, args.max_per_test_timeout)
