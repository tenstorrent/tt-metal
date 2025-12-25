
export temporigbranch=brosko/test_glx_umd
# export tempcommit=("05c84f0" "d8687fd" "3299505")
# export tempcommit=("1ce581b")
# export tempcommit=("5f72bff" "8a63e7f" "f21759e" "2646d30" "6862db7")
export allcommits=("72cd7112" "af52aa16" "bc14a3b5" "f865ace3" "82e7f387" "45742f8d" "c582ec5b" "fb6ce5c9" "78677932" "14ac4416" "0a30b1e2" "c9fc7294" "03055725" "67a67152" "d86b83ac" "30cc18fd" "8b24e259" "2568ba11" "0e071c64" "cba7b617" "646dbd58" "ba4ab7b5" "7aa315ea" "d0936031" "a780c582" "5a6d9f71" "77cfe2d8" "9e68b60e" "2a923097" "6711cedc" "8e2877c6" "9cb38471" "4aeef666" "9e6bfc61" "70db3062" "ae043aaa" "159e3004" "cb9b4f55" "10059405" "bdcda51e" "dab66912" "d2d07866" "58e5f0ea" "ad06de83" "2e824226" "ea62cf52" "b61cf119" )
# Now take every 7th commit
export tempcommit=("${allcommits[@]:0:7}")
export tempbranch=brosko/test_glx_umd_

# Declare associative array to store commit hash -> run URL mapping
declare -A run_links

# Loop over the array
for i in "${!tempcommit[@]}"; do
  # Construct branch name using tempbranch and index
  branch_name="${tempbranch}${tempcommit[$i]}"
  commit_hash="${tempcommit[$i]}"

  # Checkout in tt_metal
  git checkout "$temporigbranch"
  git checkout -b "$branch_name"
  cd tt_metal/third_party/umd

  # Checkout in umd using the corresponding commit
  git checkout "${tempcommit[$i]}"

  # Return to the root directory
  cd ../../../

  git add .
  git commit -m"umd ${tempcommit[$i]}"
  git push --set-upstream origin "$branch_name"

  # Trigger the workflow
  gh workflow run "galaxy-unit-tests.yaml" --ref "$branch_name" --repo "tenstorrent/tt-metal"

  # Wait a moment for the run to be created
  sleep 10

  # Get the run ID and URL
  run_id=$(gh run list --workflow "galaxy-unit-tests.yaml" --branch "$branch_name" --repo "tenstorrent/tt-metal" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)

  if [ -n "$run_id" ] && [ "$run_id" != "null" ]; then
    run_url="https://github.com/tenstorrent/tt-metal/actions/runs/$run_id"
    run_links["$commit_hash"]="$run_url"
    echo "Captured run for commit $commit_hash: $run_url"
  else
    run_links["$commit_hash"]="NOT_FOUND"
    echo "Warning: Could not find run for commit $commit_hash"
  fi
done

# Print all commit hash and run link pairs at the end
echo ""
echo "=== Commit Hash -> Run Link Mapping ==="
for commit_hash in "${!run_links[@]}"; do
  echo "$commit_hash ${run_links[$commit_hash]}"
done

# git checkout -b brosko/test_tghang_pre
# cd tt_metal/third_party/umd
# git pull
# git checkout brosko/test_tghang_pre
# cd ../../../
# git add .
# git commit -m'umd brosko/test_tghang_pre'
# git push --set-upstream origin brosko/test_tghang_pre
# ttmetcitg

# last change 364e45c1 Reimplement UMD BAR0 usage (#959)
# Generated using:
# git log --oneline
# 72cd7112 (HEAD -> main, origin/main, origin/HEAD) Add multicast support to TlbWindow (#1718)
# af52aa16 Remove get_register_address dead code (#1725)
# bc14a3b5 fix/Asserting on correct Blackhole firmware version check (18.5.0) (#1720)
# f865ace3 Mock device (#1705)
# 82e7f387 Fix multicast for Blackhole simulation (#1713)
# 45742f8d `umd_common` header library cleanup (#1701)
# c582ec5b Merged tt_version with semver_t (#1484)
# fb6ce5c9 Unify SPDX headers (#1645)
# 78677932 Use public ttsim for tests (#1712)
# 14ac4416 Host spec gathering script (#1627)
# 0a30b1e2 Implement read/write with reconfigure in TlbWindow (#1660)
# c9fc7294 Remove all lite fabric related code (#1683)
# 03055725 Change cached TLB size in LocalChip (#1675)
# 67a67152 Make ERISC FW hash check optional (#1711)
# d86b83ac Add ERISC FW 7.2.0 for CMFW 19.4.1 (#1710)
# 30cc18fd Read logical remote eth id for all BH systems (#1682)
# 8b24e259 Improve soc. desc paths for tests (#1703)
# 2568ba11 Make routing firmware check not throw exception (#1697)
# 0e071c64 Additional python interface for exalens (#1678)
# cba7b617 Add wh-erisc version 7.3.0 for fw 19.4.0 (#1704)
# 646dbd58 Driver version mismatch removal (#1700)
# ba4ab7b5 Remove BAR4 mapping from PCI device (#1684)
# 7aa315ea Add unity build support to UMD (#1693)
# d0936031 Calculate WH-ERISC Board ID instead of reading it (#1691)
# a780c582 Add BH-ERISC expected version 1.7.1 for FW bundle 19.3.0 (#1685)
# 5a6d9f71 Add precommit to check for periods in comments (#1646)
# 77cfe2d8 Verify ERISC FW hash in TopologyDiscovery (#1613)
# 9e68b60e Add TopologyDiscovery discover method (#1654)
# 2a923097 Fix DMA2 tests (#1681)
# 6711cedc Add standalone ASIO library (#1674)
# 8e2877c6 Fix BH p300 machine runs (#1672)
# 9cb38471 Introduce static lib (#1662)
# 4aeef666 Fix RemoteChip in case of 0 host channels (#1679)
# 9e6bfc61 Fix PCIe DMA read and write functions (#1676)
# 70db3062 Remove leftover deprecated arc_msg implementation (#1658)
# ae043aaa Use 1MB for WH cached TLB (#1669)
# 159e3004 Change EXPECT to ASSERT in sysmem tests (#1670)
# cb9b4f55 Remove unused variable from TlbManager (#1661)
# 10059405 Remove tt::umd when unnecessary (#1663)
# bdcda51e Fix TSAN reporting for RobustMutex (#1649)
# dab66912 Set expected ETH FW version according to FW bundle version (#1655)
# d2d07866 Integrate TT-KMD lib to code for TLBs (#1560)
# 58e5f0ea Change log level from warning to info for firmware check (#1657)
# ad06de83 Fix TSAN data race for remote communication (#1651)
# 2e824226 Fix Tsan/Asan python builds (#1652)
# ea62cf52 Rename ttsim to ttsim-private (#1650)
# b61cf119 (tag: v0.6.0) Version bump for changes to support TLBs from KMD (#1653)
