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

    if ($line =~ /per iteration/) {
        my @parts = split(' ', $line);
        my $us = $parts[8];
        my $index = index($parts[8], ".");
        $us = substr($us, 0, $index + 3);
        $data->[$j][$i] = $us;
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
