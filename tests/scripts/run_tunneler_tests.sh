#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    echo "Must provide TT_METAL_SLOW_DISPATCH_MODE in environment" 1>&2
    exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

export TT_METAL_CLEAR_L1=1

echo "Running tunneler tests now...";

run_test() {
    echo $1
    $1
    echo
};

run_test_with_watcher() {
    echo $1
    TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 $1
    echo
};

main() {
    # Parse the arguments
    while [[ $# -gt 0 ]]; do
      case $1 in
        --machine-type)
          machine_type=$2
          shift
          ;;
        *)
          echo "Unknown option: $1"
          exit 1
          ;;
      esac
      shift
    done

    if [[ $ARCH_NAME == "wormhole_b0" && $machine_type != "N150" ]]; then
      for max_packet_size_words in 256 512 1024 2048; do
        run_test "./build/test/tt_metal/perf_microbenchmark/routing/test_vc_uni_tunnel  --tx_x 4 --tx_y 7 --mux_x 0 --mux_y 7 --demux_x 0 --demux_y 0 --rx_x 0 --rx_y 1 --max_packet_size_words $max_packet_size_words --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 1 --data_kb_per_tx 1048576 --tunneler_queue_size_bytes 32768 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 65536"
        run_test "./build/test/tt_metal/perf_microbenchmark/routing/test_vc_bi_tunnel_2ep  --tx_x 4 --tx_y 7 --mux_x 0 --mux_y 7 --demux_x 0 --demux_y 0 --rx_x 0 --rx_y 1 --max_packet_size_words $max_packet_size_words --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 1 --data_kb_per_tx 1048576 --tunneler_queue_size_bytes 32768 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 65536"
        run_test "./build/test/tt_metal/perf_microbenchmark/routing/test_vc_bi_tunnel_4ep --tx_x 4 --tx_y 7 --mux_x 0 --mux_y 7 --demux_x 0 --demux_y 0 --rx_x 0 --rx_y 1 --max_packet_size_words $max_packet_size_words --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 1 --data_kb_per_tx 1048576 --tunneler_queue_size_bytes 16384 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 65536"
      done
    fi

}

main "$@"
