#!/bin/bash
INTERACTIVE_MODE=0
if [ $# -ge 1 ];
then
    INTERACTIVE_MODE=1
fi

pushd `pwd`
if [[ -z $TT_METAL_HOME ]];
then
    echo "TT_METAL_HOME required"
    exit -1
fi
cd ${TT_METAL_HOME}/docs/
for i in `find ./source/ -type d -name 'sweeps' -prune -o -iname '*.rst'`;
do
    echo "Checking $i"
    if [ $INTERACTIVE_MODE -gt 0 ];
    then
        aspell -c $i --personal `pwd`/aspell-dictionary.pws
    else
        X=$(cat $i | aspell --personal `pwd`/aspell-dictionary.pws -a  | grep ^\& | cut -d':' -f1  | wc -l)
        if [ $X -ne 0 ];
        then
            if [ -s $i ];
            then
                echo "-------------------------------"
                echo "There are typos in the file: $i"
                echo "Please update text in $i, or update personal dictionary as case maybe"
                echo "-------------------------------"
                exit -1
            else
                echo "Skipping empty file $i"
            fi
        fi
    fi
done
popd
