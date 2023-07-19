#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

rm log/analytical_model.log

# echo "read" >> log/analytical_model.log
# for buffer_pow in {6..18}
# do
# for transaction_pow in {6..13}
# do
# if (($buffer_pow >= $transaction_pow))
# then

# buffer=$(power_of_2 $buffer_pow)
# transaction=$(power_of_2 $transaction_pow)

# python3 profile_scripts/analytical_model.py --read-or-write read --pre-issue-overhead 17 --NIU-programming 6 --non-NIU-programming 43 --round-trip-latency 96 --flit-latency 1.01 --transfer-size $transaction --buffer-size $buffer >> log/analytical_model.log

# fi
# done
# done

echo "write" >> log/analytical_model.log
for t in {1..4}
do
for b in {1..16}
do

transaction=$((t * 2048))
buffer=$((b * transaction))

python3 profile_scripts/analytical_model.py --read-or-write write --pre-issue-overhead 12 --NIU-programming 6 --non-NIU-programming 37 --round-trip-latency 94 --flit-latency 1.01 --transfer-size $transaction --buffer-size $buffer >> log/analytical_model.log

done
done
