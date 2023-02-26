function count_loc {
    dir="$1"  
    
    shift 
    exclude_dirs="$*"
    exclude_option=""
    if [ -n "$exclude_dirs" ]; then
        exclude_option="--exclude-dir=$exclude_dirs"
    fi

    #echo "$exclude_option"
    echo -n "$dir \t$exclude_dirs \t"
    cloc --quiet "$dir" $exclude_option | awk '/SUM:/{print $5}'
}

# count_loc <dir> <exclude_dirs> # exclude_dirs comma separated
# code 

# core
echo "Core code:"
count_loc build_kernels_for_riscv tests
count_loc common
count_loc hlkc meow_hash
count_loc hostdevcommon
count_loc ll_buda tests,op_library
count_loc ll_buda_bindings
count_loc llrt tests
echo
echo

# kernels
echo "Kernels/OPs:"
echo "Kernels/OPs:"
count_loc kernels
count_loc ll_buda/op_library
count_loc programming_examples # this is small -- move to tests?
echo
echo

# device
echo "Device:"
count_loc device
echo 
echo

# infra
echo "Infra:"
count_loc git_hooks
count_loc reg_scripts
count_loc release
echo
echo

# tests
echo "Tests:"
count_loc ll_buda/tests
count_loc build_kernels_for_riscv/tests
count_loc llrt/tests
count_loc python_api_testing
echo 
echo


# external libraries
# only one file so it breaks
echo "External libraries:"
count_loc hlkc/meow_hash
echo
echo
