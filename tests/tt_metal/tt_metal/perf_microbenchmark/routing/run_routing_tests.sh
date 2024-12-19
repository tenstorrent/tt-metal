#!/bin/bash

# sanity test
echo "==== test_tx_rx ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_tx_rx

echo "==== test_mux_demux ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_mux_demux --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072

echo "==== test_mux_demux_2level ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_mux_demux_2level --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072

echo "==== test_vc_mux_demux ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_vc_mux_demux --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072

echo "==== test_vc_uni_tunnel ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_vc_uni_tunnel --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072

echo "==== test_vc_loopback_tunnel ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_vc_loopback_tunnel --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072

echo "==== test_vc_bi_tunnel_2ep ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_vc_bi_tunnel_2ep --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072

echo "==== test_vc_bi_tunnel_4ep ===="
./build/test/tt_metal/perf_microbenchmark/routing/test_vc_bi_tunnel_4ep --max_packet_size_words 2048 --tx_skip_pkt_content_gen 1 --rx_disable_data_check 1 --rx_disable_header_check 1 --tx_pkt_dest_size_choice 1 --check_txrx_timeout 0 --data_kb_per_tx 1048576 --tx_queue_size_bytes 65536 --rx_queue_size_bytes 131072 --mux_queue_size_bytes 65536 --demux_queue_size_bytes 131072
