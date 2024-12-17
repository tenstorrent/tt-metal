#!/usr/bin/perl

use strict;
use warnings;

my $file = 'log';

open my $info, $file or die "Could not open $file: $!";

my $data;

my $i = 0;
my $j = 0;
my $maxj = 0;

while (my $line = <$info>) {
    if ($line =~ /###/) {
        $i++;
        if ($j > $maxj) {
            $maxj = $j;
        }
        $j = 0;
    }

    # output line from test_pgm_dispatch
    if ($line =~ /us per iteration/) {
        my @parts = split(' ', $line);
        my $us = $parts[8];
        my $index = index($parts[8], ".");
        my $digits = index($parts[8], "us") - $index;
        $digits = $digits >= 3 ? 3 : $digits;
        $us = substr($us, 0, $index + $digits);
        $data->[$j][$i] = $us;
        $j++;
    }

    # output line from test_bw_and_latency
    if ($line =~ /BW:/) {
        my @parts = split(' ', $line);
        my $bw = $parts[7];
        $data->[$j][$i] = $bw;
        $j++;
    }

    # output latency from test_bw_and_latency
    if ($line =~ /Latency:/) {
        my @parts = split(' ', $line);
        my $bw = $parts[7];
        $data->[$j][$i] = $bw;
        $j++;
    }

    # output bw from test_rw_buffer
    if ($line =~ /Best/) {
        my @parts = split(' ', $line);
        my $bw = $parts[8];
        $data->[$j][$i] = $bw;
        $j++;
    }
}

for (my $y = 0; $y < $maxj; $y++) {
    my $eol = 0;
    for (my $x = 0; $x <= $i; $x++) {
        if (exists($data->[$y][$x])) {
            print $data->[$y][$x], ", ";
            $eol = 1;
        }
    }
    if ($eol) {
        print "\n";
    }
}

close $info;
